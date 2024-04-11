#dna pretrain
import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import editdistance
from torchsummary import summary
from torchcrf import CRF
from numpy import random
import pickle
import scipy.stats
import math

from arnie.pfunc import pfunc
from arnie.free_energy import free_energy
from arnie.bpps import bpps
from arnie.mfe import mfe
import arnie.utils as utils
os.environ['CUDA_VISIBLE_DEVICES']='0'
learning_rate=4.0e-4
Batch_size=1
Conv_kernel=7
dropout=0.3
embedding_dim=128
num_encoder_layers=4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)
import fm
# Load RNA-FM model
fm_pretrain_model, fm_pretrain_alphabet = fm.pretrained.rna_fm_t12()
fm_pretrain_batch_converter = fm_pretrain_alphabet.get_batch_converter()
fm_pretrain_model=fm_pretrain_model.to(device)


tokens = 'ACGU().BEHIMSXDF'   #D start,F end
vocab_size=len(tokens)

SEED=4
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True



patience=50
error_alpha=0.5
error_beta=5
epochs=1000
nhead=4
nStrDim=8
Use_pretrain_model=True

class RNADataset(Dataset):
    def __init__(self,seqs,seqsOri, Stru,Loop,As):
        if Use_pretrain_model:
            self.seqs=seqsOri
        else:
            self.seqs=seqs

        self.Stru = Stru
        self.Loop = Loop
        self.As=As
        self.length=len(self.As)

    def __getitem__(self,idx):
        
        return self.seqs[idx], self.Stru[idx],self.Loop[idx],self.As[idx]

    def __len__(self):
        return self.length

def preprocess_inputs(np_seq):

    re_seq=[]
    for i in range(len(np_seq)):

        re_seq.append([tokens.index(s) for s in np_seq[i]])

    re_seq=np.array(re_seq)


    return re_seq


def get_structure_adj(seq_length,structure,sequence):
    
    
    
    cue = []
    a_structures = {
            ("A", "U"): np.zeros([seq_length, seq_length]),
            ("C", "G"): np.zeros([seq_length, seq_length]),
            ("U", "G"): np.zeros([seq_length, seq_length]),
            ("U", "A"): np.zeros([seq_length, seq_length]),
            ("G", "C"): np.zeros([seq_length, seq_length]),
            ("G", "U"): np.zeros([seq_length, seq_length]),
    }

    for i in range(seq_length):
        if structure[i] == "(":
            cue.append(i)
        elif structure[i] == ")":
            start = cue.pop()
            a_structures[(sequence[start], sequence[i])][start, i] = 1
            a_structures[(sequence[i], sequence[start])][i, start] = 1

    a_strc = np.stack([a for a in a_structures.values()], axis=2)


    a_strc = np.sum(a_strc, axis=2, keepdims=True)


    a_strc = np.array(a_strc,dtype='float32')

    '''new = np.zeros((Ss.shape[0], Ss.shape[1] + 2, Ss.shape[2] + 2, Ss.shape[3]))
    new[:, 1:-1, 1:-1, :] = Ss'''

    return a_strc



    

class Model(nn.Module):
    def __init__(self,vocab_size,embedding_dim,pred_dim,dropout,nhead,num_encoder_layers):
        super().__init__()

        if Use_pretrain_model:
            self.embeddingSeq_fm = nn.Sequential(nn.Linear(640, 320),
                                       nn.LayerNorm(320),
                                       nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(320 , embedding_dim)
                                       )
            self.embeddingSeq_dnapt2 = nn.Sequential(nn.Linear(768, 256),
                                       nn.LayerNorm(256),
                                       nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(256 , embedding_dim)
                                       )
        else:
            self.embeddingSeq=nn.Embedding(vocab_size,embedding_dim)

        #self.embeddingStru = nn.Embedding(vocab_size, embedding_dim)
        self.embeddingloop = nn.Embedding(vocab_size, embedding_dim)


        self.embeddingstr = nn.Sequential(nn.Conv2d(nStrDim, embedding_dim, 1),
                                        nn.BatchNorm2d(embedding_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(dropout),
                                        nn.AdaptiveAvgPool2d((None, 1))
                                        )
        '''self.fc_fusion = nn.Sequential(nn.Linear(embedding_dim * 3, embedding_dim),
                                       nn.LayerNorm(embedding_dim),
                                       nn.ReLU(),
                                       nn.Linear(embedding_dim, embedding_dim)
                                       )'''
        
        encoder_layer_share = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_encoder_share = nn.TransformerEncoder(encoder_layer_share, num_layers=num_encoder_layers)


        encoder_layer_seq_fm = nn.TransformerEncoderLayer(d_model=embedding_dim , nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_encoder_seq_fm = nn.TransformerEncoder(encoder_layer_seq_fm, num_layers=num_encoder_layers)
        self.conv_encoder_seq_fm=nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim, Conv_kernel),
                                            nn.BatchNorm1d(num_features=embedding_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout),

                                            nn.ConvTranspose1d(embedding_dim , embedding_dim, Conv_kernel),
                                            nn.BatchNorm1d(num_features=embedding_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout),
                                            )
        
        

        encoder_layer_loop = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_encoder_loop = nn.TransformerEncoder(encoder_layer_loop, num_layers=num_encoder_layers)
        self.conv_encoder_loop = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim, Conv_kernel),
                                            nn.BatchNorm1d(num_features=embedding_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout),

                                            nn.ConvTranspose1d(embedding_dim , embedding_dim, Conv_kernel),
                                            nn.BatchNorm1d(num_features=embedding_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout),
                                            )

        encoder_layer_str = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_encoder_str = nn.TransformerEncoder(encoder_layer_str, num_layers=num_encoder_layers)
        self.conv_encoder_str = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim, Conv_kernel),
                                            nn.BatchNorm1d(num_features=embedding_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout),

                                            nn.ConvTranspose1d(embedding_dim , embedding_dim, Conv_kernel),
                                            nn.BatchNorm1d(num_features=embedding_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout),
                                            )

        encoder_layer_fustrans = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,dropout=dropout,batch_first=True)
        self.transformer_encoder_fustrans = nn.TransformerEncoder(encoder_layer_fustrans, num_layers=num_encoder_layers)
        self.conv_encoder_fusconv = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim*2, Conv_kernel),
                                            nn.BatchNorm1d(num_features=embedding_dim*2),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout),

                                            

                                            nn.ConvTranspose1d(embedding_dim*2, embedding_dim, Conv_kernel),
                                            nn.BatchNorm1d(num_features=embedding_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout),
                                            )




        self.conv_encoder_finfus = nn.Sequential(nn.Conv1d(embedding_dim*2, embedding_dim*4, Conv_kernel),
                                            nn.BatchNorm1d(num_features=embedding_dim*4),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout),

                                            

                                            nn.ConvTranspose1d(embedding_dim*4, embedding_dim*2, Conv_kernel),
                                            nn.BatchNorm1d(num_features=embedding_dim*2),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout),
                                            )
        self.pre=nn.Sequential(nn.Linear(embedding_dim*2, embedding_dim),
                                       nn.LayerNorm(embedding_dim),
                                       nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(embedding_dim , pred_dim)
                                       )
        
        
        

    def forward(self,stru,loop,As,seq=None,fm_seq=None):
        #As shape [64, 107, 107, 8]

        if not Use_pretrain_model:
            embeddedSeq=self.embeddingSeq(seq)
            
            embeddedLoop = self.embeddingloop(loop)
        
            As=self.embeddingstr(As.permute(0,3, 1, 2) ).permute(0, 2, 3, 1)  #[64, 107, 1, 128]
            As = torch.squeeze(As)  #[64, 107, 128]
        
            embeddedSeq=self.encoder_share(embeddedSeq)
            embedded_seq1=self.transformer_encoder_seq(embeddedSeq)
            embedded_seq2 = self.conv_encoder_seq(embeddedSeq.permute(0,2,1)).permute(0,2,1)
        
            embeddedLoop = self.encoder_share(embeddedLoop)
            embedded_loop1 = self.transformer_encoder_loop(embeddedLoop)
            embedded_loop2 = self.conv_encoder_loop(embeddedLoop.permute(0,2,1)).permute(0,2,1)

            As = self.encoder_share(As)
            embedded_str1 = self.transformer_encoder_str(As)
            embedded_str2 = self.conv_encoder_str(As.permute(0,2,1)).permute(0,2,1)
        
            embedded_fus_trans =embedded_seq1+embedded_loop1+embedded_str1
            embedded_fus_conv = embedded_seq2 + embedded_loop2 + embedded_str2

            del embedded_seq1,embedded_loop1,embedded_str1,embedded_seq2 , embedded_loop2 , embedded_str2

            embedded_fus_trans = self.transformer_encoder_fustrans(embedded_fus_trans)
            embedded_fus_conv=self.conv_encoder_fusconv(embedded_fus_conv.permute(0,2,1)).permute(0,2,1)

            embedded_cat = torch.cat((embedded_fus_trans, embedded_fus_conv), 2)
            pre_out=self.pre(embedded_cat)

            return pre_out
        else:
            embeddedSeq_fm=self.embeddingSeq_fm(fm_seq)
            
            
            

        
            embeddedLoop = self.embeddingloop(loop)
        
            As=self.embeddingstr(As.permute(0,3, 1, 2) ).permute(0, 2, 3, 1)  #[64, 107, 1, 128]
            As = torch.squeeze(As)  #[64, 107, 128]
        
            embeddedSeq_fm_share=self.transformer_encoder_share(embeddedSeq_fm)
            embedded_seq_fm1=self.transformer_encoder_seq_fm(embeddedSeq_fm_share)
            embedded_seq_fm2 = self.conv_encoder_seq_fm(embeddedSeq_fm_share.permute(0,2,1)).permute(0,2,1)

            
        
            embeddedLoop= self.transformer_encoder_share(embeddedLoop)
            embedded_loop1 = self.transformer_encoder_loop(embeddedLoop)
            embedded_loop2 = self.conv_encoder_loop(embeddedLoop.unsqueeze(0).permute(0,2,1)).permute(0,2,1)

            As = self.transformer_encoder_share(As)
            embedded_str1 = self.transformer_encoder_str(As)
            embedded_str2 = self.conv_encoder_str(As.unsqueeze(0).permute(0,2,1)).permute(0,2,1)
        
            embedded_fus_trans =embedded_seq_fm1+embedded_loop1+embedded_str1
            embedded_fus_conv = embedded_seq_fm2 + embedded_loop2 + embedded_str2

            del embedded_loop1,embedded_str1, embedded_loop2 , embedded_str2,embedded_seq_fm1,embedded_seq_fm2

            embedded_fus_trans = self.transformer_encoder_fustrans(embedded_fus_trans)
            embedded_fus_conv=self.conv_encoder_fusconv(embedded_fus_conv.permute(0,2,1)).permute(0,2,1)

            embedded_cat = torch.cat((embedded_fus_trans, embedded_fus_conv), 2)
            embedded_cat=self.conv_encoder_finfus(embedded_cat.permute(0,2,1)).permute(0,2,1)
            pre_out=self.pre(embedded_cat)

            

            
            return pre_out







def get_distance_matrix(As):
    idx = np.arange(As.shape[0])
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1 / Ds
    
    

    Dss = []
    for i in [1, 2, 4]:
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis=2)
    return Ds

def calc_neighbor(d, dim, n):
    lst_x,lst_y = np.where(d==n)
    for c, x in enumerate(lst_x):
        y = lst_y[c]
        if x+1<dim:
            d[x+1,y] = min(d[x+1,y], n+1)
        if y+1<dim:
            d[x,y+1] = min(d[x,y+1], n+1)
        if x-1>=0:
            d[x-1,y] = min(d[x-1,y], n+1)
        if y-1>=0:
            d[x,y-1] = min(d[x,y-1], n+1)
    return d
def get_distance_matrix_2d(Ss):
    
    
    dim = Ss.shape[0]
    
    s = Ss[ :, :, 0]
    d = 10 + np.zeros_like(s)
    d[s == 1] = 1
    for i in range(dim):
        d[i, i] = 0
    for x in range(0, 9):
        d = calc_neighbor(d, dim, x)
        
    d = np.array(d) + 1
    d = 1 / d
    d = d[ :, :, None]

    Dss = []
    for i in [1, 2, 4]:
        Dss.append(d ** i)
    Ds = np.stack(Dss, axis=2)
    return Ds[:, :, :, 0]
def get_As(testSeqLen, testStruOri, testSeqOri):
    testStruAdj = get_structure_adj(testSeqLen, testStruOri, testSeqOri)
    
    
    test_bp_matrix=bpps(testSeqOri, package='vienna_2').tolist()
    

    As_test = np.array(test_bp_matrix)
    Ds_test = get_distance_matrix(As_test)
    DDs_test = get_distance_matrix_2d(testStruAdj)
    
    As_test = np.concatenate([As_test[ :, :, None], testStruAdj, Ds_test, DDs_test], axis=2).astype(np.float32)
    
    del testStruAdj, Ds_test, DDs_test
    return As_test

def train_fold():
    
    

    
    testDataFrame=pd.read_csv('./data/GSE173083_188_withstrloop.csv')
    print("testData shape:",testDataFrame.shape)
    
    testSeqLen = np.array(testDataFrame['RNA length'].values.tolist())
    testSeqOri = np.array(testDataFrame['RNA sequence'].values.tolist())
    testStruOri = np.array(testDataFrame['structure'].values.tolist())
    testLoopOri = np.array(testDataFrame['loop'].values.tolist())
    for i in range(len(testSeqLen)):
        if testSeqLen[i]>1022:
            testSeqLen[i]=1022
            testSeqOri[i]=testSeqOri[i][:1022]
            testStruOri[i]=testStruOri[i][:1022]
            testLoopOri[i]=testLoopOri[i][:1022]
    

    testSeq = preprocess_inputs(testSeqOri)
    testStru = preprocess_inputs(testStruOri)
    testLoop = preprocess_inputs(testLoopOri)
    
    print('testSeq shape:', testSeq.shape)
    
    
    As_test=[]
    for i in range(len(testSeqOri)):
        temp=get_As(testSeqLen[i], testStruOri[i], testSeqOri[i])
        As_test.append(temp)
        
   
    dataset_test = RNADataset(testSeq, testSeqOri,testStru,testLoop,As_test)
    dataloader_test = DataLoader(dataset_test, batch_size=Batch_size, shuffle=False)
    

    model=Model(vocab_size,embedding_dim,5,dropout,nhead,num_encoder_layers).to(device)
    model.load_state_dict(torch.load('./best_model_weights_withoutDNABERT'))
    model.eval()
    model_output=[]

    
    for batch_idx, data in enumerate(dataloader_test, 0):
        

        seqs,strus,loops,As=data

        if not Use_pretrain_model :
            seqs=seqs.to(device)
        else:
            fm_pretrain_model.eval()
            seqs_pretrainformat_fm = []
            seqs_dnapt2_id=[]
                
            for temp_i in range(len(seqs)):
                seqs_pretrainformat_fm.append(("RNA", seqs[temp_i]))
                
                    
            _, _, fm_pre_batch_tokens = fm_pretrain_batch_converter(seqs_pretrainformat_fm)
            with torch.no_grad():
                fm_pre_results = fm_pretrain_model(fm_pre_batch_tokens.to(device), repr_layers=[12])
            fm_seqs = fm_pre_results["representations"][12][:, 1:-1, :]

            
                    

        strus = torch.tensor(strus).to(device)
        loops = torch.tensor(loops).to(device)
        
        As = torch.tensor(As).to(device)
        if not Use_pretrain_model :
            pre_out = model(strus,loops,As,seq=seqs)
        else:
            pre_out = model(strus,loops,As,fm_seq=fm_seqs)
        model_output.append(pre_out[:, :,:3].squeeze().tolist())

    
    pickle.dump(model_output, open( "mRNA_deg_spe_cor_modelpre.p", "wb" ) )
    model_output=pickle.load( open( "mRNA_deg_spe_cor_modelpre.p", "rb" ) )
    

def test():
    model_output=pickle.load( open( "mRNA_deg_spe_cor_modelpre.p", "rb" ) )
    testDataFrame=pd.read_csv('./data/GSE173083_188_withstrloop.csv')
    test_half_life = np.array(testDataFrame['In-solution degradation coefficient'].values.tolist())
    test_half_life=test_half_life.tolist()
    model_output_reactivity=[]
    model_output_deg_Mg_pH10=[]
    model_output_deg_Mg_50C=[]
    for i in range(len(model_output)):
        temp_reactivity=0
        temp_deg_Mg_pH10=0
        temp_deg_Mg_50C=0
        for j in range(len(model_output[i])):
            temp_reactivity=temp_reactivity+model_output[i][j][0]
            temp_deg_Mg_pH10=temp_deg_Mg_pH10+model_output[i][j][1]
            temp_deg_Mg_50C=temp_deg_Mg_50C+model_output[i][j][2]
        
        temp_reactivity=temp_reactivity/len(model_output[i])
        temp_deg_Mg_pH10=temp_deg_Mg_pH10/len(model_output[i])
        temp_deg_Mg_50C=temp_deg_Mg_50C/len(model_output[i])

        model_output_reactivity.append(temp_reactivity)
        model_output_deg_Mg_pH10.append(temp_deg_Mg_pH10)
        model_output_deg_Mg_50C.append(temp_deg_Mg_50C)

    print(scipy.stats.spearmanr(model_output_reactivity, test_half_life))
    print(scipy.stats.spearmanr(model_output_deg_Mg_pH10, test_half_life))
    print(scipy.stats.spearmanr(model_output_deg_Mg_50C, test_half_life))
    
        

             
        
#train_fold()
test()

