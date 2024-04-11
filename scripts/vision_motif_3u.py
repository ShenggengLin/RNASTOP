# Import the required libraries
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
from arnie.pfunc import pfunc
from arnie.free_energy import free_energy
from arnie.bpps import bpps
from arnie.mfe import mfe
import arnie.utils as utils

#Fixed random seeds to ensure reproducibility of the results
seed = 4
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

#The hyperparameters of the model can be modified here
os.environ['CUDA_VISIBLE_DEVICES']='0'
learning_rate=4.0e-4
Batch_size=512
Conv_kernel=7
dropout=0.3
embedding_dim=128
num_encoder_layers=4
patience=50
error_alpha=0.5
error_beta=5
epochs=1000
nhead=4
nStrDim=4
Use_pretrain_model=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

# Load RNA-FM model
import fm
fm_pretrain_model, fm_pretrain_alphabet = fm.pretrained.rna_fm_t12()
fm_pretrain_batch_converter = fm_pretrain_alphabet.get_batch_converter()
fm_pretrain_model=fm_pretrain_model.to(device)



#The mRNA sequences are clustered, and each mRNA is assigned a weight according to the clustering results
def sample_weight(text_lists,thred=20):
    clusters=[]
    for text in text_lists:
        flag=False
        for cluster in clusters:
            for _text in cluster:
                if editdistance.distance(text,_text)<=thred:
                    cluster.add(text)
                    flag = True
                    break
            if flag:
                break
        if not flag:
            clusters.append({text})

    #valid_clusters=[i for i in clusters if len(i)>50]
    print('total cluster:{}'.format(len(clusters)))

    clusters_sam_wei=[]
    for i in range(len(clusters)):
        clusters_sam_wei.append(1/np.sqrt(len(clusters[i])))

    sam_wei=[]
    for text in text_lists:
        for j in range(len(clusters)):
            if text in clusters[j]:
                sam_wei.append(clusters_sam_wei[j])
                break
    sam_wei=np.array(sam_wei)

    return sam_wei




#Definition the word list
tokens = 'ACGU().BEHIMSXDF'   #D start,F end
vocab_size=len(tokens)




# Definition the RNADataset
class RNADataset(Dataset):
    def __init__(self,seqs,seqsOri, As,train_sam_wei=None,train_error_weights=None,sam_aug_flag=None):
        if Use_pretrain_model:
            self.seqs=seqsOri
        else:
            self.seqs=seqs

        
        
        
        self.As=As
        self.train_sam_wei = train_sam_wei
        self.train_error_weights=train_error_weights
        self.sam_aug_flag=sam_aug_flag
        self.length=len(seqs)
        

    def __getitem__(self,idx):
        if (self.train_sam_wei is not None) and (self.train_error_weights is not None) and (self.sam_aug_flag is not None):
            return self.seqs[idx],self.As[idx], self.train_sam_wei[idx],self.train_error_weights[idx],self.sam_aug_flag[idx]
        else:
            return self.seqs[idx],  self.As[idx]

    def __len__(self):
        return self.length

#Get the embedding of the mrna sequences
def preprocess_inputs(np_seq):

    re_seq=[]
    for i in range(len(np_seq)):

        re_seq.append([tokens.index(s) for s in np_seq[i]])

    re_seq=np.array(re_seq)


    return re_seq

#Get the adjacency matrix of the mrna
def get_structure_adj(data_seq_length,data_structure,data_sequence):
    Ss = []
    for i in (range(len(data_sequence))):
        seq_length = data_seq_length[i]
        structure = data_structure[i]
        sequence = data_sequence[i]

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




        Ss.append(a_strc)

    Ss = np.array(Ss,dtype='float32')

    '''new = np.zeros((Ss.shape[0], Ss.shape[1] + 2, Ss.shape[2] + 2, Ss.shape[3]))
    new[:, 1:-1, 1:-1, :] = Ss'''

    return Ss


    
# Define the RNADegpre model
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
            
        else:
            self.embeddingSeq=nn.Embedding(vocab_size,embedding_dim)

        #self.embeddingStru = nn.Embedding(vocab_size, embedding_dim)
        


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
        
        
        

    def forward(self,As,seq=None,fm_seq=None,dnapt2_seq=None):
        #As shape [64, 107, 107, 8]

        if not Use_pretrain_model:
            embeddedSeq=self.embeddingSeq(seq)
            
            
        else:
            embeddedSeq_fm=self.embeddingSeq_fm(fm_seq)
            
            
            

        
            #embeddedLoop = self.embeddingloop(loop)
        
            As=self.embeddingstr(As.permute(0,3, 1, 2) ).permute(0, 2, 3, 1)  #[64, 107, 1, 128]
            As = torch.squeeze(As)  #[64, 107, 128]
            
            
        
            embeddedSeq_fm_share=self.transformer_encoder_share(embeddedSeq_fm)
            embedded_seq_fm1=self.transformer_encoder_seq_fm(embeddedSeq_fm_share)
            embedded_seq_fm2 = self.conv_encoder_seq_fm(embeddedSeq_fm_share.permute(0,2,1)).permute(0,2,1)

           
            '''embeddedLoop= self.transformer_encoder_share(embeddedLoop)
            embedded_loop1 = self.transformer_encoder_loop(embeddedLoop)
            embedded_loop2 = self.conv_encoder_loop(embeddedLoop.permute(0,2,1)).permute(0,2,1)'''

            As = self.transformer_encoder_share(As)
            embedded_str1 = self.transformer_encoder_str(As)
            embedded_str2 = self.conv_encoder_str(As.permute(0,2,1)).permute(0,2,1)
        
            embedded_fus_trans =embedded_seq_fm1+embedded_str1
            embedded_fus_conv = embedded_seq_fm2+embedded_str2

            del embedded_str1, embedded_str2,embedded_seq_fm1,embedded_seq_fm2

            embedded_fus_trans = self.transformer_encoder_fustrans(embedded_fus_trans)
            embedded_fus_conv=self.conv_encoder_fusconv(embedded_fus_conv.permute(0,2,1)).permute(0,2,1)

            embedded_cat = torch.cat((embedded_fus_trans, embedded_fus_conv), 2)
            embedded_cat=self.conv_encoder_finfus(embedded_cat.permute(0,2,1)).permute(0,2,1)
            pre_out=self.pre(embedded_cat)

            

            
            return pre_out

# Define different data types
pred_cols_train = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
pred_cols_test = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

# In the process of training the model, different weights are assigned to different data types
LOSS_WGTS = [0.3, 0.3, 0.3, 0.05, 0.05]
pred_cols_errors=['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_Mg_50C','deg_error_pH10','deg_error_50C']


    




#get the 1-dimensional and 2-dimensional distance matrix of mRNA
def get_distance_matrix(As):
    idx = np.arange(As.shape[1])
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1 / Ds
    Ds = Ds[None, :, :]
    Ds = np.repeat(Ds, len(As), axis=0)

    Dss = []
    for i in [1, 2, 4]:
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis=3)
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
    Ds = []
    n = Ss.shape[0]
    dim = Ss.shape[1]
    for i in range(n):
        s = Ss[i, :, :, 0]
        d = 10 + np.zeros_like(s)
        d[s == 1] = 1
        for i in range(dim):
            d[i, i] = 0
        for x in range(0, 9):
            d = calc_neighbor(d, dim, x)
        Ds.append(d)
    Ds = np.array(Ds) + 1
    Ds = 1 / Ds
    Ds = Ds[:, :, :, None]

    Dss = []
    for i in [1, 2, 4]:
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis=3)
    return Ds[:, :, :, :, 0]

#The codons are sorted according to their degradability, and the degradability value and the location of the codon are returned
def max_deg_pro(pre_list,geshu):
    val=[]
    for i in range(len(pre_list)//3):
        val.append(pre_list[i*3]+pre_list[i*3+1]+pre_list[i*3+2])

    sorted_id = sorted(range(len(val)), key=lambda k: val[k], reverse=True)
    sorted_id=sorted_id[:geshu]
    vale=[]
    for i in range(len(sorted_id)):
        vale.append(val[sorted_id[i]])

    return vale,sorted_id

#Returns the most easily degraded codon
def max_deg_mimazi(seq,sort_id):
    mimazi=[]
    for i in range(len(sort_id)):
        mimazi.append(seq[sort_id[i]]+seq[sort_id[i]+1]+seq[sort_id[i]+2])
    return mimazi

#Define the codon table
mimazi_table=[{'GCU','GCC','GCA','GCG'},{'CGU','CGC','CGA','CGG','AGA','AGG'},{'AAU','AAC'},{'GAU','GAC'},{'UGU','UGC'},
              {'CAA','CAG'},{'GAA','GAG'},{'GGU','GGC','GGA','GGG'},{'CAU','CAC'},{'AUU','AUC','AUA'},
              {'UUA','UUG','CUU','CUC','CUA','CUG'},{'AAA','AAG'},{'AUG'},{'UUU','UUC'},{'CCU','CCC','CCA','CCG'},
              {'UCU','UCC','UCA','UCG','AGU','AGC'},{'ACU','ACC','ACA','ACG'},{'UGG'},{'UAU','UAC'},{'GUU','GUC','GUA','GUG'}]

#Return the mutated sequence
def seq_opti(seq,sorted_id):
    #sorted_id [91, 81, 145]

    seq_after_tubian1=[]
    seq_after_tubian2=[]
    seq_after_tubian3=[]
    
    seq_ori_charlist=list(seq)
    
    
    mimazi_ori=max_deg_mimazi(seq,sorted_id)  #['CCG', 'GUC', 'CCG']
    mimazi_can_choice=[]
    for i in range(len(mimazi_ori)):
        now_mimazi=mimazi_ori[i]
        for j in range(len(mimazi_table)):
            if now_mimazi in mimazi_table[j]:
                mimazi_can_choice.append(mimazi_table[j])

    def get_new_seq(seq_ori_charlist,id1,mimazi_opt1,id2,mimazi_opt2,id3,mimazi_opt3):
            seq_ori_charlist[id1]=mimazi_opt1[0]
            seq_ori_charlist[id1+1]=mimazi_opt1[1]
            seq_ori_charlist[id1+2]=mimazi_opt1[2]

            seq_ori_charlist[id2]=mimazi_opt2[0]
            seq_ori_charlist[id2+1]=mimazi_opt2[1]
            seq_ori_charlist[id2+2]=mimazi_opt2[2]

            seq_ori_charlist[id3]=mimazi_opt3[0]
            seq_ori_charlist[id3+1]=mimazi_opt3[1]
            seq_ori_charlist[id3+2]=mimazi_opt3[2]

            seq_now=''.join(seq_ori_charlist)
            return seq_now

    
        
    for temp_mimazi0 in mimazi_can_choice[0]:
        for temp_mimazi1 in mimazi_can_choice[1]:
            for temp_mimazi2 in mimazi_can_choice[2]:

                if (mimazi_ori[0]!= temp_mimazi0) and (mimazi_ori[1]== temp_mimazi1) and (mimazi_ori[2]== temp_mimazi2):
                    new_seq=get_new_seq(seq_ori_charlist,sorted_id[0],temp_mimazi0,sorted_id[1],temp_mimazi1,sorted_id[2],temp_mimazi2)
                    seq_after_tubian1.append(new_seq)
                if (mimazi_ori[0]== temp_mimazi0) and (mimazi_ori[1]!= temp_mimazi1) and (mimazi_ori[2]== temp_mimazi2):
                    new_seq=get_new_seq(seq_ori_charlist,sorted_id[0],temp_mimazi0,sorted_id[1],temp_mimazi1,sorted_id[2],temp_mimazi2)
                    seq_after_tubian1.append(new_seq)
                if (mimazi_ori[0]== temp_mimazi0) and (mimazi_ori[1]== temp_mimazi1) and (mimazi_ori[2]!= temp_mimazi2):
                    new_seq=get_new_seq(seq_ori_charlist,sorted_id[0],temp_mimazi0,sorted_id[1],temp_mimazi1,sorted_id[2],temp_mimazi2)
                    seq_after_tubian1.append(new_seq)

                if (mimazi_ori[0]!= temp_mimazi0) and (mimazi_ori[1]!= temp_mimazi1) and (mimazi_ori[2]== temp_mimazi2):
                    new_seq=get_new_seq(seq_ori_charlist,sorted_id[0],temp_mimazi0,sorted_id[1],temp_mimazi1,sorted_id[2],temp_mimazi2)
                    seq_after_tubian2.append(new_seq)
                if (mimazi_ori[0]!= temp_mimazi0) and (mimazi_ori[1]== temp_mimazi1) and (mimazi_ori[2]!= temp_mimazi2):
                    new_seq=get_new_seq(seq_ori_charlist,sorted_id[0],temp_mimazi0,sorted_id[1],temp_mimazi1,sorted_id[2],temp_mimazi2)
                    seq_after_tubian2.append(new_seq)
                if (mimazi_ori[0]== temp_mimazi0) and (mimazi_ori[1]!= temp_mimazi1) and (mimazi_ori[2]!= temp_mimazi2):
                    new_seq=get_new_seq(seq_ori_charlist,sorted_id[0],temp_mimazi0,sorted_id[1],temp_mimazi1,sorted_id[2],temp_mimazi2)
                    seq_after_tubian2.append(new_seq)
                    
                if (mimazi_ori[0]!= temp_mimazi0) and (mimazi_ori[1]!= temp_mimazi1) and (mimazi_ori[2]!= temp_mimazi2):
                    new_seq=get_new_seq(seq_ori_charlist,sorted_id[0],temp_mimazi0,sorted_id[1],temp_mimazi1,sorted_id[2],temp_mimazi2)
                    seq_after_tubian3.append(new_seq)
        
    print('number of tubian1:',len(seq_after_tubian1))
    print('number of tubian2:',len(seq_after_tubian2))
    print('number of tubian3:',len(seq_after_tubian3))
    return seq_after_tubian1,seq_after_tubian2,seq_after_tubian3
    
    




def motif_3u_seq_discovery(motif_3u_seq):
    
    
    motif_3u_seq_ori=np.array(motif_3u_seq)
    motif_3u_seq=preprocess_inputs(motif_3u_seq_ori)
    print("seq num:",len(motif_3u_seq))
    
    seq_len=[]
    for i in range(len(motif_3u_seq)):
        seq_len.append(len(motif_3u_seq[i]))

    bp_matrix = []
    for test_seq in motif_3u_seq_ori:
        
        bp_matrix.append(bpps(test_seq, package='vienna_2').tolist())
    bp_matrix = np.array(bp_matrix)

    As = np.array(bp_matrix)
    
    Ds = get_distance_matrix(As)
    
    As = np.concatenate([As[:, :, :, None],  Ds], axis=3).astype(np.float32)
   
    del  Ds

    dataset = RNADataset(motif_3u_seq, motif_3u_seq_ori,As)
    dataloader = DataLoader(dataset, batch_size=Batch_size, shuffle=False)
    

    model=Model(vocab_size,embedding_dim,5,dropout,nhead,num_encoder_layers).to(device)
    model.load_state_dict(torch.load('./best_model_weights_withoutstrloopdnabert'))
    model.eval()


    pre_out_sum = [0,0,0,0,0]
    pre_out_sum=np.array(pre_out_sum)

    
    for batch_idx, data in enumerate(dataloader, 0):
        

        seqs,As=data

        
        fm_pretrain_model.eval()
        seqs_pretrainformat_fm = []
            
                
        for temp_i in range(len(seqs)):
            seqs_pretrainformat_fm.append(("RNA", seqs[temp_i]))
                
                    
        _, _, fm_pre_batch_tokens = fm_pretrain_batch_converter(seqs_pretrainformat_fm)
        with torch.no_grad():
            fm_pre_results = fm_pretrain_model(fm_pre_batch_tokens.to(device), repr_layers=[12])
        fm_seqs = fm_pre_results["representations"][12][:, 1:-1, :]

            
                    
        
        As = As.to(device)
            
        
        
        pre_out = model(As,fm_seq=fm_seqs)
        
        pre_out_ph10=pre_out.cpu()[:,20:25,1].tolist()  #取ph10
        pre_out_ph10=np.array(pre_out_ph10)
        pre_out_ph10=pre_out_ph10.sum(axis=0)

        pre_out_sum=pre_out_sum+pre_out_ph10
    
    pre_out_sum_avg=np.divide(pre_out_sum,len(motif_3u_seq))
    
    return pre_out_sum_avg.tolist()
    
    
        
motif_3u_seq=[]
f_3umotif_seq=open("./data/3u_motif_seq.txt")
for seq in open("./data/3u_motif_seq.txt"):
    motif_3u_seq.append(seq.replace('\n','').replace('\r',''))
print("num of total seq:",len(motif_3u_seq))

A_20pos=[]
A_21pos=[]
A_22pos=[]
A_23pos=[]
A_24pos=[]
C_20pos=[]
C_21pos=[]
C_22pos=[]
C_23pos=[]
C_24pos=[]
G_20pos=[]
G_21pos=[]
G_22pos=[]
G_23pos=[]
G_24pos=[]
U_20pos=[]
U_21pos=[]
U_22pos=[]
U_23pos=[]
U_24pos=[]
for i in range(len(motif_3u_seq)):
    if motif_3u_seq[i][20]=='A':
        A_20pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][21]=='A':
        A_21pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][22]=='A':
        A_22pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][23]=='A':
        A_23pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][24]=='A':
        A_24pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][20]=='C':
        C_20pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][21]=='C':
        C_21pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][22]=='C':
        C_22pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][23]=='C':
        C_23pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][24]=='C':
        C_24pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][20]=='G':
        G_20pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][21]=='G':
        G_21pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][22]=='G':
        G_22pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][23]=='G':
        G_23pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][24]=='G':
        G_24pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][20]=='U':
        U_20pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][21]=='U':
        U_21pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][22]=='U':
        U_22pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][23]=='U':
        U_23pos.append(motif_3u_seq[i])
    if motif_3u_seq[i][24]=='U':
        U_24pos.append(motif_3u_seq[i])
#motif_3u_seq_discovery()

final_preout=[]
final_preout.append(motif_3u_seq_discovery(A_20pos))
final_preout.append(motif_3u_seq_discovery(A_21pos))
final_preout.append(motif_3u_seq_discovery(A_22pos))
final_preout.append(motif_3u_seq_discovery(A_23pos))
final_preout.append(motif_3u_seq_discovery(A_24pos))
final_preout.append(motif_3u_seq_discovery(C_20pos))
final_preout.append(motif_3u_seq_discovery(C_21pos))
final_preout.append(motif_3u_seq_discovery(C_22pos))
final_preout.append(motif_3u_seq_discovery(C_23pos))
final_preout.append(motif_3u_seq_discovery(C_24pos))
final_preout.append(motif_3u_seq_discovery(G_20pos))
final_preout.append(motif_3u_seq_discovery(G_21pos))
final_preout.append(motif_3u_seq_discovery(G_22pos))
final_preout.append(motif_3u_seq_discovery(G_23pos))
final_preout.append(motif_3u_seq_discovery(G_24pos))
final_preout.append(motif_3u_seq_discovery(U_20pos))
final_preout.append(motif_3u_seq_discovery(U_21pos))
final_preout.append(motif_3u_seq_discovery(U_22pos))
final_preout.append(motif_3u_seq_discovery(U_23pos))
final_preout.append(motif_3u_seq_discovery(U_24pos))
min_final_preout=final_preout[0][0]
max_final_preout=final_preout[0][0]
for i in range(len(final_preout)):
    for j in range(len(final_preout[i])):
        if final_preout[i][j]<min_final_preout:
            min_final_preout=final_preout[i][j]
        if final_preout[i][j]>max_final_preout:
            max_final_preout=final_preout[i][j]

for i in range(len(final_preout)):
    for j in range(len(final_preout[i])):
        final_preout[i][j]=(final_preout[i][j]-min_final_preout)/(max_final_preout-min_final_preout)
final_preout=np.array(final_preout)
print(final_preout)
