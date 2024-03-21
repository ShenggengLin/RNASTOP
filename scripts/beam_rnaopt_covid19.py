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
import copy

from arnie.pfunc import pfunc
from arnie.free_energy import free_energy
from arnie.bpps import bpps
from arnie.mfe import mfe
import arnie.utils as utils
seed = 4
beam_sear_geshu=10
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES']='1'
learning_rate=4.0e-4
Batch_size=64
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





tokens = 'ACGU().BEHIMSXDF'   #D start,F end
vocab_size=len(tokens)





patience=50
error_alpha=0.5
error_beta=5
epochs=1000
nhead=4
nStrDim=4
Use_pretrain_model=True

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

def preprocess_inputs(np_seq):

    re_seq=[]
    for i in range(len(np_seq)):

        re_seq.append([tokens.index(s) for s in np_seq[i]])

    re_seq=np.array(re_seq)


    return re_seq


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
            As = torch.unsqueeze(As,0)
            
        
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


pred_cols_train = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
pred_cols_test = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

LOSS_WGTS = [0.3, 0.3, 0.3, 0.05, 0.05]
pred_cols_errors=['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_Mg_50C','deg_error_pH10','deg_error_50C']


    





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

def max_deg_mimazi(seq,sort_id):
    mimazi=[]
    for i in range(len(sort_id)):
        mimazi.append(seq[sort_id[i]*3]+seq[sort_id[i]*3+1]+seq[sort_id[i]*3+2])
    return mimazi

mimazi_table=[{'GCU','GCC','GCA','GCG'},{'CGU','CGC','CGA','CGG','AGA','AGG'},{'AAU','AAC'},{'GAU','GAC'},{'UGU','UGC'},
              {'CAA','CAG'},{'GAA','GAG'},{'GGU','GGC','GGA','GGG'},{'CAU','CAC'},{'AUU','AUC','AUA'},
              {'UUA','UUG','CUU','CUC','CUA','CUG'},{'AAA','AAG'},{'AUG'},{'UUU','UUC'},{'CCU','CCC','CCA','CCG'},
              {'UCU','UCC','UCA','UCG','AGU','AGC'},{'ACU','ACC','ACA','ACG'},{'UGG'},{'UAU','UAC'},{'GUU','GUC','GUA','GUG'}]
'''
def seq_opti(seq,sorted_id):
    
    seq_now=list(seq)
    mimazi=max_deg_mimazi(seq,sorted_id)
    mimazi_opt=[]
    for i in range(len(mimazi)):
        now_mimazi=mimazi[i]
        for j in range(len(mimazi_table)):
            if now_mimazi in mimazi_table[j]:
                if mimazi_table[j]==1:
                    mimazi_opt.append(now_mimazi)
                    break
                else:
                    random_mimazi=random.choice(list(mimazi_table[j]-{now_mimazi}))
                    mimazi_opt.append(random_mimazi)
                    break
    
    for i in range(len(mimazi_opt)):
        seq_now[sorted_id[i]]=mimazi_opt[i][0]
        seq_now[sorted_id[i]+1]=mimazi_opt[i][1]
        seq_now[sorted_id[i]+2]=mimazi_opt[i][2]
    seq_now=''.join(seq_now)
    
    print(mimazi)
    print(mimazi_opt)
    return seq_now
'''

def seq_opti(seq_ori,sorted_id,seq_ori_mfe):
    #sorted_id [91, 81, 145]

    seq_after_tubian=[]
    seq_after_tubian_mfe=[]
    seq_ori_charlist=list(seq_ori)
    
    
    mimazi_ori=max_deg_mimazi(seq_ori,sorted_id)  #['CCG', 'GUC', 'CCG']
    mimazi_can_choice=[]
    for i in range(len(mimazi_ori)):
        now_mimazi=mimazi_ori[i]
        for j in range(len(mimazi_table)):
            if now_mimazi in mimazi_table[j]:
                mimazi_can_choice.append(mimazi_table[j])

    def get_new_seq(seq_ori_charlist,id,mimazi_opt):
            seq_ori_charlist_temp=copy.deepcopy(seq_ori_charlist)
            seq_ori_charlist_temp[id*3]=mimazi_opt[0]
            seq_ori_charlist_temp[id*3+1]=mimazi_opt[1]
            seq_ori_charlist_temp[id*3+2]=mimazi_opt[2]
            seq_now=''.join(seq_ori_charlist_temp)
            return seq_now

    for i in range(len(mimazi_ori)):
        flag_nomut_best=1
        for temp_mimazi in mimazi_can_choice[i]:
            if mimazi_ori[i]!= temp_mimazi:
                new_seq=get_new_seq(seq_ori_charlist,sorted_id[i],temp_mimazi)
                temp_mfe=free_energy(new_seq, package='vienna_2')
                if temp_mfe<seq_ori_mfe:
                    seq_after_tubian.append(new_seq)
                    seq_after_tubian_mfe.append(temp_mfe)
                    flag_nomut_best=0
        if flag_nomut_best==1:
            seq_after_tubian.append(seq_ori)
            seq_after_tubian_mfe.append(seq_ori_mfe)
    
    

    sorted_mfe = sorted(range(len(seq_after_tubian_mfe)), key=lambda k: seq_after_tubian_mfe[k])
    sorted_mfe=sorted_mfe[:beam_sear_geshu]
    final_seq=[]
    final_score=[]
    for i in range(len(sorted_mfe)):
        final_seq.append(seq_after_tubian[sorted_mfe[i]])
        final_score.append(seq_after_tubian_mfe[sorted_mfe[i]])
    
    return final_seq,final_score

def beam_search_single(seq_ori,len_seq_fenge,seq_ori_mfe):
    
    seq_ori_len=len(seq_ori)
    
    pre_out_ph10=[]
    for i in range(seq_ori_len//len_seq_fenge + 1):
        seq_ori_i=np.array([seq_ori[i*len_seq_fenge:min(seq_ori_len,(i+1)*len_seq_fenge)]])

        seq_i=preprocess_inputs(seq_ori_i)
        seq_i_len=[]
        for i in range(len(seq_i)):
            seq_i_len.append(len(seq_i[i]))
    
        i_bp_matrix = []
        for test_seq in seq_ori_i:
        
            i_bp_matrix.append(bpps(test_seq, package='vienna_2').tolist())
        i_bp_matrix = np.array(i_bp_matrix)
    
        As_i = np.array(i_bp_matrix)
    
        Ds_i = get_distance_matrix(As_i)
    
        As_i = np.concatenate([As_i[:, :, :, None],  Ds_i], axis=3).astype(np.float32)
   
        del  Ds_i
        dataset_i = RNADataset(seq_i, seq_ori_i,As_i)
        dataloader_i = DataLoader(dataset_i, batch_size=Batch_size, shuffle=False)
        model=Model(vocab_size,embedding_dim,5,dropout,nhead,num_encoder_layers).to(device)
        model.load_state_dict(torch.load('./best_model_weights_withoutstrloopdnabert'))
        model.eval()
        for batch_idx, data in enumerate(dataloader_i, 0):
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
            pre_out_ph10_i=pre_out[0,:,1].tolist()  #取gE_WTph10
            pre_out_ph10=pre_out_ph10+pre_out_ph10_i
        
    vale,sorted_id=max_deg_pro(pre_out_ph10,beam_sear_geshu)
    mimazi=max_deg_mimazi(seq_ori,sorted_id)
    final_seq,final_score=seq_opti(seq_ori,sorted_id,seq_ori_mfe)

    return final_seq,final_score
    
    

def beam_search_multi(seq_list,score_list):
    seq_multi=[]
    score_multi=[]
    for i in range(len(seq_list)):
        final_seq_single,final_score_single=beam_search_single(seq_list[i],999,score_list[i])
        seq_multi.extend(final_seq_single)
        score_multi.extend(final_score_single)

    sorted_mfe_multi = sorted(range(len(score_multi)), key=lambda k: score_multi[k])
    sorted_mfe_multi=sorted_mfe_multi[:beam_sear_geshu]
    final_seq_multi=[]
    final_score_multi=[]
    for i in range(len(sorted_mfe_multi)):
        final_seq_multi.append(seq_multi[sorted_mfe_multi[i]])
        final_score_multi.append(score_multi[sorted_mfe_multi[i]])
    return final_seq_multi,final_score_multi

    


#seq_ori_COVID19_BNT=["AUGGGGACAGUUAAUAAACCAUCGGGGUUCAAAUCGGGCGGA"]
seq_ori_COVID19_BNT=["AUGUUCGUGUUCCUGGUGCUGCUGCCUCUGGUGUCCAGCCAGUGUGUGAACCUGACCACCAGAACACAGCUGCCUCCAGCCUACACCAACAGCUUUACCAGAGGCGUGUACUACCCCGACAAGGUGUUCAGAUCCAGCGUGCUGCACUCUACCCAGGACCUGUUCCUGCCUUUCUUCAGCAACGUGACCUGGUUCCACGCCAUCCACGUGUCCGGCACCAAUGGCACCAAGAGAUUCGACAACCCCGUGCUGCCCUUCAACGACGGGGUGUACUUUGCCAGCACCGAGAAGUCCAACAUCAUCAGAGGCUGGAUCUUCGGCACCACACUGGACAGCAAGACCCAGAGCCUGCUGAUCGUGAACAACGCCACCAACGUGGUCAUCAAAGUGUGCGAGUUCCAGUUCUGCAACGACCCCUUCCUGGGCGUCUACUACCACAAGAACAACAAGAGCUGGAUGGAAAGCGAGUUCCGGGUGUACAGCAGCGCCAACAACUGCACCUUCGAGUACGUGUCCCAGCCUUUCCUGAUGGACCUGGAAGGCAAGCAGGGCAACUUCAAGAACCUGCGCGAGUUCGUGUUUAAGAACAUCGACGGCUACUUCAAGAUCUACAGCAAGCACACCCCUAUCAACCUCGUGCGGGAUCUGCCUCAGGGCUUCUCUGCUCUGGAACCCCUGGUGGAUCUGCCCAUCGGCAUCAACAUCACCCGGUUUCAGACACUGCUGGCCCUGCACAGAAGCUACCUGACACCUGGCGAUAGCAGCAGCGGAUGGACAGCUGGUGCCGCCGCUUACUAUGUGGGCUACCUGCAGCCUAGAACCUUCCUGCUGAAGUACAACGAGAACGGCACCAUCACCGACGCCGUGGAUUGUGCUCUGGAUCCUCUGAGCGAGACAAAGUGCACCCUGAAGUCCUUCACCGUGGAAAAGGGCAUCUACCAGACCAGCAACUUCCGGGUGCAGCCCACCGAAUCCAUCGUGCGGUUCCCCAAUAUCACCAAUCUGUGCCCCUUCGGCGAGGUGUUCAAUGCCACCAGAUUCGCCUCUGUGUACGCCUGGAACCGGAAGCGGAUCAGCAAUUGCGUGGCCGACUACUCCGUGCUGUACAACUCCGCCAGCUUCAGCACCUUCAAGUGCUACGGCGUGUCCCCUACCAAGCUGAACGACCUGUGCUUCACAAACGUGUACGCCGACAGCUUCGUGAUCCGGGGAGAUGAAGUGCGGCAGAUUGCCCCUGGACAGACAGGCAAGAUCGCCGACUACAACUACAAGCUGCCCGACGACUUCACCGGCUGUGUGAUUGCCUGGAACAGCAACAACCUGGACUCCAAAGUCGGCGGCAACUACAAUUACCUGUACCGGCUGUUCCGGAAGUCCAAUCUGAAGCCCUUCGAGCGGGACAUCUCCACCGAGAUCUAUCAGGCCGGCAGCACCCCUUGUAACGGCGUGGAAGGCUUCAACUGCUACUUCCCACUGCAGUCCUACGGCUUUCAGCCCACAAAUGGCGUGGGCUAUCAGCCCUACAGAGUGGUGGUGCUGAGCUUCGAACUGCUGCAUGCCCCUGCCACAGUGUGCGGCCCUAAGAAAAGCACCAAUCUCGUGAAGAACAAAUGCGUGAACUUCAACUUCAACGGCCUGACCGGCACCGGCGUGCUGACAGAGAGCAACAAGAAGUUCCUGCCAUUCCAGCAGUUUGGCCGGGAUAUCGCCGAUACCACAGACGCCGUUAGAGAUCCCCAGACACUGGAAAUCCUGGACAUCACCCCUUGCAGCUUCGGCGGAGUGUCUGUGAUCACCCCUGGCACCAACACCAGCAAUCAGGUGGCAGUGCUGUACCAGGACGUGAACUGUACCGAAGUGCCCGUGGCCAUUCACGCCGAUCAGCUGACACCUACAUGGCGGGUGUACUCCACCGGCAGCAAUGUGUUUCAGACCAGAGCCGGCUGUCUGAUCGGAGCCGAGCACGUGAACAAUAGCUACGAGUGCGACAUCCCCAUCGGCGCUGGAAUCUGCGCCAGCUACCAGACACAGACAAACAGCCCUCGGAGAGCCAGAAGCGUGGCCAGCCAGAGCAUCAUUGCCUACACAAUGUCUCUGGGCGCCGAGAACAGCGUGGCCUACUCCAACAACUCUAUCGCUAUCCCCACCAACUUCACCAUCAGCGUGACCACAGAGAUCCUGCCUGUGUCCAUGACCAAGACCAGCGUGGACUGCACCAUGUACAUCUGCGGCGAUUCCACCGAGUGCUCCAACCUGCUGCUGCAGUACGGCAGCUUCUGCACCCAGCUGAAUAGAGCCCUGACAGGGAUCGCCGUGGAACAGGACAAGAACACCCAAGAGGUGUUCGCCCAAGUGAAGCAGAUCUACAAGACCCCUCCUAUCAAGGACUUCGGCGGCUUCAAUUUCAGCCAGAUUCUGCCCGAUCCUAGCAAGCCCAGCAAGCGGAGCUUCAUCGAGGACCUGCUGUUCAACAAAGUGACACUGGCCGACGCCGGCUUCAUCAAGCAGUAUGGCGAUUGUCUGGGCGACAUUGCCGCCAGGGAUCUGAUUUGCGCCCAGAAGUUUAACGGACUGACAGUGCUGCCUCCUCUGCUGACCGAUGAGAUGAUCGCCCAGUACACAUCUGCCCUGCUGGCCGGCACAAUCACAAGCGGCUGGACAUUUGGAGCAGGCGCCGCUCUGCAGAUCCCCUUUGCUAUGCAGAUGGCCUACCGGUUCAACGGCAUCGGAGUGACCCAGAAUGUGCUGUACGAGAACCAGAAGCUGAUCGCCAACCAGUUCAACAGCGCCAUCGGCAAGAUCCAGGACAGCCUGAGCAGCACAGCAAGCGCCCUGGGAAAGCUGCAGGACGUGGUCAACCAGAAUGCCCAGGCACUGAACACCCUGGUCAAGCAGCUGUCCUCCAACUUCGGCGCCAUCAGCUCUGUGCUGAACGAUAUCCUGAGCAGACUGGACaaagUgGAGGCCGAGGUGCAGAUCGACAGACUGAUCACAGGCAGACUGCAGAGCCUCCAGACAUACGUGACCCAGCAGCUGAUCAGAGCCGCCGAGAUUAGAGCCUCUGCCAAUCUGGCCGCCACCAAGAUGUCUGAGUGUGUGCUGGGCCAGAGCAAGAGAGUGGACUUUUGCGGCAAGGGCUACCACCUGAUGAGCUUCCCUCAGUCUGCCCCUCACGGCGUGGUGUUUCUGCACGUGACAUAUGUGCCCGCUCAAGAGAAGAAUUUCACCACCGCUCCAGCCAUCUGCCACGACGGCAAAGCCCACUUUCCUAGAGAAGGCGUGUUCGUGUCCAACGGCACCCAUUGGUUCGUGACACAGCGGAACUUCUACGAGCCCCAGAUCAUCACCACCGACAACACCUUCGUGUCUGGCAACUGCGACGUCGUGAUCGGCAUUGUGAACAAUACCGUGUACGACCCUCUGCAGCCCGAGCUGGACAGCUUCAAAGAGGAACUGGACAAGUACUUUAAGAACCACACAAGCCCCGACGUGGACCUGGGCGAUAUCAGCGGAAUCAAUGCCAGCGUCGUGAACAUCCAGAAAGAGAUCGACCGGCUGAACGAGGUGGCCAAGAAUCUGAACGAGAGCCUGAUCGACCUGCAAGAACUGGGGAAGUACGAGCAGUACAUCAAGUGGCCCUGGUACAUCUGGCUGGGCUUUAUCGCCGGACUGAUUGCCAUCGUGAUGGUCACAAUCAUGCUGUGUUGCAUGACCAGCUGCUGUAGCUGCCUGAAGGGCUGUUGUAGCUGUGGCAGCUGCUGCAAGUUCGACGAGGACGAUUCUGAGCCCGUGCUGAAGGGCGUGAAACUGCACUACACA"]
seq_ori_COVID19_BNT[0]=seq_ori_COVID19_BNT[0].upper()

best_mfe=free_energy(seq_ori_COVID19_BNT[0], package='vienna_2')
best_seq=seq_ori_COVID19_BNT[0]
print('seq_ori_mfe:')
print(best_mfe)
score_ori_list=[best_mfe]

final_seq_multi,final_score_multi=beam_search_multi(seq_ori_COVID19_BNT,score_ori_list)
seq_lujing=[]
score_lujing=[]
while min(final_score_multi) <best_mfe:
    
    seq_lujing.append(final_seq_multi)
    score_lujing.append(final_score_multi)
    print(final_score_multi[0])
    print(final_seq_multi[0])
    for i in range(len(best_seq)//3):
        if (best_seq[i*3]+best_seq[i*3+1]+best_seq[i*3+2])!=(final_seq_multi[0][i*3]+final_seq_multi[0][i*3+1]+final_seq_multi[0][i*3+2]):
            print(i+1)
            print(best_seq[i*3]+best_seq[i*3+1]+best_seq[i*3+2])
            print(final_seq_multi[0][i*3]+final_seq_multi[0][i*3+1]+final_seq_multi[0][i*3+2])
    
    best_mfe=final_score_multi[0]
    best_seq=final_seq_multi[0]
    final_seq_multi,final_score_multi=beam_search_multi(final_seq_multi,final_score_multi)

gE_WT_opt_score=pd.DataFrame(score_lujing)
gE_WT_opt_score.to_csv('./covid19_opt_score_ifpanduan.csv',encoding='gbk')




