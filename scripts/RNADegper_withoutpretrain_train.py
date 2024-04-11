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

#The hyperparameters of the model can be modified here
os.environ['CUDA_VISIBLE_DEVICES']='0'
learning_rate=4.0e-4
Batch_size=64
Conv_kernel=7
dropout=0.3
embedding_dim=128
num_encoder_layers=4
patience=50
error_alpha=0.5
error_beta=5
epochs=1000
nhead=4
nStrDim=8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

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

#Fixed random seeds to ensure reproducibility of the results
SEED = 4
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# Definition the RNADataset
class RNADataset(Dataset):
    def __init__(self,seqs,seqsOri, Stru,Loop,labels,As,train_sam_wei=None,train_error_weights=None,sam_aug_flag=None):
        
        self.seqs=seqs

        self.Stru = Stru
        self.Loop = Loop
        self.labels=labels
        self.As=As
        self.train_sam_wei = train_sam_wei
        self.train_error_weights=train_error_weights
        self.sam_aug_flag=sam_aug_flag
        self.length=len(self.labels)

    def __getitem__(self,idx):
        if (self.train_sam_wei is not None) and (self.train_error_weights is not None) and (self.sam_aug_flag is not None):
            return self.seqs[idx],self.Stru[idx],self.Loop[idx],self.labels[idx], self.As[idx], self.train_sam_wei[idx],self.train_error_weights[idx],self.sam_aug_flag[idx]
        else:
            return self.seqs[idx], self.Stru[idx],self.Loop[idx],self.labels[idx], self.As[idx]

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
        
        
        

    def forward(self,stru,loop,As,seq):
        #As shape [64, 107, 107, 8]

        
        
        embeddedSeq=self.embeddingSeq(seq)
            
            
            

        
        embeddedLoop = self.embeddingloop(loop)
        
        As=self.embeddingstr(As.permute(0,3, 1, 2) ).permute(0, 2, 3, 1)  #[64, 107, 1, 128]
        As = torch.squeeze(As)  #[64, 107, 128]
        
        embeddedSeq_share=self.transformer_encoder_share(embeddedSeq)
        embedded_seq1=self.transformer_encoder_seq_fm(embeddedSeq_share)
        embedded_seq2 = self.conv_encoder_seq_fm(embeddedSeq_share.permute(0,2,1)).permute(0,2,1)

            
        
        embeddedLoop= self.transformer_encoder_share(embeddedLoop)
        embedded_loop1 = self.transformer_encoder_loop(embeddedLoop)
        embedded_loop2 = self.conv_encoder_loop(embeddedLoop.permute(0,2,1)).permute(0,2,1)

        As = self.transformer_encoder_share(As)
        embedded_str1 = self.transformer_encoder_str(As)
        embedded_str2 = self.conv_encoder_str(As.permute(0,2,1)).permute(0,2,1)
        
        embedded_fus_trans =embedded_seq1+embedded_loop1+embedded_str1
        embedded_fus_conv = embedded_seq2 + embedded_loop2 + embedded_str2

        del embedded_loop1,embedded_str1, embedded_loop2 , embedded_str2,embedded_seq1,embedded_seq2

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

# Defined loss function
def mean_squared(y_true, y_pred):

    return torch.mean(torch.sqrt(torch.mean((y_true-y_pred)**2, axis=1))) # a number =each sample site
def MCRMSE(y_pred,y_true):
    y_true = torch.where(torch.isnan(y_true), y_pred, y_true)

    s = mean_squared(y_true[:, :, 0], y_pred[:, :, 0]) / 1.0
    s = s+mean_squared(y_true[:, :, 1], y_pred[:, :, 1]) / 1.0
    s = s + mean_squared(y_true[:, :, 2], y_pred[:, :, 2]) / 1.0
    s=s/3.0
    return s
def mean_squared_sw_tew(y_true, y_pred, sample_weight,exp_err):
    temp=torch.sqrt(torch.mean(exp_err*((y_true-y_pred)**2), axis=1))
    return torch.sum(temp * sample_weight) / torch.sum(sample_weight)  # a number =each sample site
def MCRMSE_NAN_SW_TEW(y_pred,y_true,sam_weig=None,tew=None):
    y_true = torch.where(torch.isnan(y_true), y_pred, y_true)
    if sam_weig==None:
        sam_weig=torch.tensor(np.array([1.0 for _ in range(len(y_pred))],dtype='float32')).to(device)
    if tew==None:
        tew=torch.ones_like(y_true).to(device)
    s = (mean_squared_sw_tew(y_true[:, :, 0], y_pred[:, :, 0], sample_weight=sam_weig, exp_err=tew[:, :, 0]) /1.0) * LOSS_WGTS[0]
    
    for i in range(1, len(pred_cols_train)):
        s += (mean_squared_sw_tew(y_true[:, :, i], y_pred[:, :, i], sample_weight=sam_weig, exp_err=tew[:, :, i]) /1.0) * LOSS_WGTS[i]
    return s
    
# The data is preprocessed and nan is used to replace the data that does not meet the filtering conditions
def filter_train(df):
    for i in range(len(df)):
        if df['SN_filter'][i]==0:

            index1 = set([j for j, x in enumerate(df['reactivity_error'][i]) if x > 10])
            index2 = set([j for j, x in enumerate(df['reactivity'][i]) if x < -1.5])
            index = list(index1 & index2)
            for k in index:
                df['reactivity'][i][k] = np.nan

            index1 = set([j for j, x in enumerate(df['deg_error_Mg_pH10'][i]) if x > 10])
            index2 = set([j for j, x in enumerate(df['deg_Mg_pH10'][i]) if x < -1.5])
            index = list(index1 & index2)
            for k in index:
                df['deg_Mg_pH10'][i][k] = np.nan

            index1 = set([j for j, x in enumerate(df['deg_error_pH10'][i]) if x > 10])
            index2 = set([j for j, x in enumerate(df['deg_pH10'][i]) if x < -1.5])
            index = list(index1 & index2)
            for k in index:
                df['deg_pH10'][i][k] = np.nan

            index1 = set([j for j, x in enumerate(df['deg_error_Mg_50C'][i]) if x > 10])
            index2 = set([j for j, x in enumerate(df['deg_Mg_50C'][i]) if x < -1.5])
            index = list(index1 & index2)
            for k in index:
                df['deg_Mg_50C'][i][k] = np.nan

            index1 = set([j for j, x in enumerate(df['deg_error_50C'][i]) if x > 10])
            index2 = set([j for j, x in enumerate(df['deg_50C'][i]) if x < -1.5])
            index = list(index1 & index2)
            for k in index:
                df['deg_50C'][i][k] = np.nan

    return df


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

# train and test model
def train_fold():
    train_pubtest_DataFrame=pd.read_json('./data/Kaggle_train.json', lines=True)

    train_pubtest_DataFrame_SN1=train_pubtest_DataFrame[train_pubtest_DataFrame['SN_filter']==1]
    train_pubtest_DataFrame_SN1=shuffle(train_pubtest_DataFrame_SN1)
    train_pubtest_DataFrame_SN0=train_pubtest_DataFrame[train_pubtest_DataFrame['SN_filter']==0]

    
    

    trainDataFrame=pd.concat([train_pubtest_DataFrame_SN1[:-400],train_pubtest_DataFrame_SN0],ignore_index=True)
    trainDataFrame=filter_train(trainDataFrame)#[:100]
    

    print('begin cluster...')
    sam_wei=sample_weight(trainDataFrame['sequence'])
    trainDataFrame['sam_wei']=sam_wei
    

    testDataFrame_pub=train_pubtest_DataFrame_SN1[-400:].reset_index(drop=True)#[:100]
    

    testDataFrame_107=pd.read_csv('./data/Kaggle_test.csv',encoding='utf-8')
    testDataFrame_107=testDataFrame_107[testDataFrame_107['seq_length']==107]#[:100]

    testDataFrame_130=pd.read_csv('./data/private_test_labels.csv',encoding='utf-8')
    testDataFrame_130=testDataFrame_130[testDataFrame_130['test_filter']==1]#[:100]

    print("trainDataFrame shape:",trainDataFrame.shape)
    print("testDataFrame_pub shape:", testDataFrame_pub.shape)
    print("testDataFrame_107 shape:", testDataFrame_107.shape)
    print("testDataFrame_130 shape:", testDataFrame_130.shape)
    

    


    train_sam_wei=trainDataFrame['sam_wei'].values.tolist()
    train_sam_wei_aug=train_sam_wei+train_sam_wei
    train_sam_wei_aug=np.array(train_sam_wei_aug,dtype='float32')
    del train_sam_wei

    train_exp_errors=np.transpose(np.array(trainDataFrame[pred_cols_errors].values.tolist(),dtype='float32'),(0,2,1))
    train_error_weights = error_alpha + np.exp(-train_exp_errors * error_beta)  ######RNAdegformer

    train_error_weights_rev=[]
    for i in range(len(train_error_weights)):
        temp=[]
        train_error_weights_I_len=len(train_error_weights[i])
        for j in range(train_error_weights_I_len):
            temp.append(train_error_weights[i][train_error_weights_I_len-j-1].tolist())
        train_error_weights_rev.append(temp)
    train_error_weights_rev =np.array(train_error_weights_rev,dtype='float32')
    train_error_weights_aug=np.vstack((train_error_weights,train_error_weights_rev))
    del train_error_weights,train_error_weights_rev


    trainSeqOri=trainDataFrame['sequence'].values.tolist()
    train_sam_ini_num=len(trainSeqOri)
    train_seq_len=len(trainSeqOri[0])
    
    
    sam_aug_flag=[0] * train_sam_ini_num + [1] * train_sam_ini_num 
    
    trainSeqOri_rev=[]
    for i in range(len(trainSeqOri)):
        trainSeqOri_rev.append(trainSeqOri[i][::-1])
    trainSeqOri_aug=np.array(trainSeqOri+trainSeqOri_rev)    #4800*
    del trainSeqOri,trainSeqOri_rev
    
    testSeqOri_pub = np.array(testDataFrame_pub['sequence'].values.tolist())
    testSeqOri_107 = np.array(testDataFrame_107['sequence'].values.tolist())
    testSeqOri_130 = np.array(testDataFrame_130['sequence'].values.tolist())
    

    trainStruOri = trainDataFrame['structure'].values.tolist()
    trainStruOri_rev = []
    for i in range(len(trainStruOri)):
        temp=trainStruOri[i][::-1]
        temp=temp.replace("(", "*")
        temp=temp.replace(")", "(")
        temp=temp.replace("*", ")")

        trainStruOri_rev.append(temp)

    trainStruOri_aug = np.array(trainStruOri + trainStruOri_rev)  # 4800*
    del trainStruOri, trainStruOri_rev

    testStruOri_pub = np.array(testDataFrame_pub['structure'].values.tolist())
    testStruOri_107 = np.array(testDataFrame_107['structure'].values.tolist())
    testStruOri_130 = np.array(testDataFrame_130['structure'].values.tolist())

    trainLoopOri = trainDataFrame['predicted_loop_type'].values.tolist()
    trainLoopOri_rev = []
    for i in range(len(trainLoopOri)):

        trainLoopOri_rev.append(trainLoopOri[i][::-1])

    trainLoopOri_aug = np.array(trainLoopOri + trainLoopOri_rev)  # 4800*
    del trainLoopOri, trainLoopOri_rev

    testLoopOri_pub = np.array(testDataFrame_pub['predicted_loop_type'].values.tolist())
    testLoopOri_107 = np.array(testDataFrame_107['predicted_loop_type'].values.tolist())
    testLoopOri_130 = np.array(testDataFrame_130['predicted_loop_type'].values.tolist())

    trainSeq_aug=preprocess_inputs(trainSeqOri_aug)
    testSeq_pub = preprocess_inputs(testSeqOri_pub)
    testSeq_107 = preprocess_inputs(testSeqOri_107)
    testSeq_130 = preprocess_inputs(testSeqOri_130)

    trainStru_aug = preprocess_inputs(trainStruOri_aug)
    testStru_pub = preprocess_inputs(testStruOri_pub)
    testStru_107 = preprocess_inputs(testStruOri_107)
    testStru_130 = preprocess_inputs(testStruOri_130)

    trainLoop_aug = preprocess_inputs(trainLoopOri_aug)
    testLoop_pub = preprocess_inputs(testLoopOri_pub)
    testLoop_107 = preprocess_inputs(testLoopOri_107)
    testLoop_130 = preprocess_inputs(testLoopOri_130)

    print('testSeq_pub shape:', testSeq_pub.shape)
    print('testSeq_107 shape:', testSeq_107.shape)
    print('testSeq_130 shape:', testSeq_130.shape)



    trainSeqLen = trainDataFrame['seq_length'].values.tolist()
    trainSeqLen_aug=trainSeqLen+trainSeqLen
    trainSeqLen_aug=np.array(trainSeqLen_aug)
    del trainSeqLen
    
    testSeqLen_pub = np.array(testDataFrame_pub['seq_length'].values.tolist())
    testSeqLen_107 = np.array(testDataFrame_107['seq_length'].values.tolist())
    testSeqLen_130 = np.array(testDataFrame_130['seq_length'].values.tolist())
    print('trainSeqLen_aug shape:', trainSeqLen_aug.shape)
    
    


    trainStruAdj_aug=get_structure_adj(trainSeqLen_aug, trainStruOri_aug, trainSeqOri_aug)
    
    testStruAdj_pub = get_structure_adj(testSeqLen_pub, testStruOri_pub, testSeqOri_pub)
    testStruAdj_107 = get_structure_adj(testSeqLen_107, testStruOri_107, testSeqOri_107)
    testStruAdj_130 = get_structure_adj(testSeqLen_130, testStruOri_130, testSeqOri_130)

    print('trainStruAdj shape:',trainStruAdj_aug.shape)   #(4800, 107, 107, 1)
    


    train_bp_matrix_aug=[]
    for train_seq in trainSeqOri_aug:
        train_bp_matrix_aug.append(bpps(train_seq, package='vienna_2').tolist())
        
    train_bp_matrix_aug =np.array(train_bp_matrix_aug)
    print('train_bp_matrix shape:',train_bp_matrix_aug.shape)   #(4800, 107, 107)
    

    test_bp_matrix_pub = []
    for test_seq in testSeqOri_pub:
        test_bp_matrix_pub.append(bpps(test_seq, package='vienna_2').tolist())
    test_bp_matrix_pub = np.array(test_bp_matrix_pub)

    test_bp_matrix_107 = []
    for test_seq in testSeqOri_107:
        test_bp_matrix_107.append(bpps(test_seq, package='vienna_2').tolist())
    test_bp_matrix_107 = np.array(test_bp_matrix_107)

    test_bp_matrix_130 = []
    for test_seq in testSeqOri_130:
        test_bp_matrix_130.append(bpps(test_seq, package='vienna_2').tolist())
    test_bp_matrix_130 = np.array(test_bp_matrix_130)

    #As_train = [train_bp_matrix]
    #As_val = [val_bp_matrix]
    #As_test = [test_bp_matrix]

    As_train_aug = np.array(train_bp_matrix_aug)
    As_test_pub = np.array(test_bp_matrix_pub)
    As_test_107 = np.array(test_bp_matrix_107)
    As_test_130 = np.array(test_bp_matrix_130)

    Ds_train_aug = get_distance_matrix(As_train_aug)
    Ds_test_pub = get_distance_matrix(As_test_pub)
    Ds_test_107 = get_distance_matrix(As_test_107)
    Ds_test_130 = get_distance_matrix(As_test_130)

    DDs_train_aug = get_distance_matrix_2d(trainStruAdj_aug)
    DDs_test_pub = get_distance_matrix_2d(testStruAdj_pub)
    DDs_test_107 = get_distance_matrix_2d(testStruAdj_107)
    DDs_test_130 = get_distance_matrix_2d(testStruAdj_130)

    As_train_aug = np.concatenate([As_train_aug[:, :, :, None], trainStruAdj_aug, Ds_train_aug, DDs_train_aug], axis=3).astype(np.float32)
    As_test_pub = np.concatenate([As_test_pub[:, :, :, None], testStruAdj_pub, Ds_test_pub, DDs_test_pub], axis=3).astype(np.float32)
    As_test_107 = np.concatenate([As_test_107[:, :, :, None], testStruAdj_107, Ds_test_107, DDs_test_107], axis=3).astype(np.float32)
    As_test_130 = np.concatenate([As_test_130[:, :, :, None], testStruAdj_130, Ds_test_130, DDs_test_130], axis=3).astype(np.float32)

    del trainStruAdj_aug, Ds_train_aug, DDs_train_aug,testStruAdj_pub, Ds_test_pub, DDs_test_pub,testStruAdj_107, Ds_test_107, DDs_test_107,testStruAdj_130, Ds_test_130, DDs_test_130

    print("As_train shape: ",As_train_aug.shape)



    train_labels = np.array(trainDataFrame[pred_cols_train].values.tolist(),dtype='float32').transpose((0, 2, 1))

    train_labels_rev = []
    for i in range(len(train_labels)):
        temp = []
        train_labels_I_len = len(train_labels[i])
        for j in range(train_labels_I_len):
            temp.append(train_labels[i][train_labels_I_len - j - 1].tolist())
        train_labels_rev.append(temp)
    train_labels_rev = np.array(train_labels_rev, dtype='float32')
    train_labels_aug = np.vstack((train_labels, train_labels_rev))
    train_seq_scored=len(train_labels_aug[0])
    
    del train_labels, train_labels_rev

    
    

    print("train_labels shape",train_labels_aug.shape)
    

    
    testData_labels_num_pub = np.array(testDataFrame_pub[pred_cols_test].values.tolist(),dtype='float32').transpose((0, 2, 1))

    testData_labels_107 = np.array(testDataFrame_107[pred_cols_test].values.tolist())
    testData_labels_130 = np.array(testDataFrame_130[pred_cols_test].values.tolist())
    testData_labels_num_107 = []
    for i in range(len(testData_labels_107)):
        sam_labels_num = []
        for j in range(len(testData_labels_107[0])):
            temp = testData_labels_107[i][j].lstrip("[").rstrip("]").split(",")
            temp2 = list(filter(lambda x: x != ' None', temp))
            sam_labels_num.append(list(map(float, temp2)))
        testData_labels_num_107.append(sam_labels_num)
    testData_labels_num_107 = np.array(testData_labels_num_107,dtype='float32').transpose((0, 2, 1))

    testData_labels_num_130 = []
    for i in range(len(testData_labels_130)):
        sam_labels_num = []
        for j in range(len(testData_labels_130[0])):
            temp = testData_labels_130[i][j].lstrip("[").rstrip("]").split(",")
            temp2 = list(filter(lambda x: x != ' None', temp))
            sam_labels_num.append(list(map(float, temp2)))
        testData_labels_num_130.append(sam_labels_num)
    testData_labels_num_130 = np.array(testData_labels_num_130,dtype='float32').transpose((0, 2, 1))

    print("testData_labels_num_pub shape",testData_labels_num_pub.shape)
    print("testData_labels_num_107 shape",testData_labels_num_107.shape)
    print("testData_labels_num_130 shape",testData_labels_num_130.shape)

    test_seq_scored_pub=len(testData_labels_num_pub[0])
    pred_dim_pub=len(testData_labels_num_pub[0][0])

    test_seq_scored_107=len(testData_labels_num_107[0])
    pred_dim_107=len(testData_labels_num_107[0][0])
    

    test_seq_scored_130=len(testData_labels_num_130[0])
    pred_dim_130=len(testData_labels_num_130[0][0])
    

    pred_dim_train=len(train_labels_aug[0][0])

    
    dataset_train = RNADataset(trainSeq_aug, trainSeqOri_aug,trainStru_aug,trainLoop_aug,train_labels_aug,As_train_aug,train_sam_wei_aug,train_error_weights_aug,sam_aug_flag)
    dataset_test_pub = RNADataset(testSeq_pub, testSeqOri_pub,testStru_pub,testLoop_pub,testData_labels_num_pub,As_test_pub)
    dataset_test_107 = RNADataset(testSeq_107, testSeqOri_107,testStru_107,testLoop_107,testData_labels_num_107,As_test_107)
    dataset_test_130 = RNADataset(testSeq_130, testSeqOri_130,testStru_130,testLoop_130,testData_labels_num_130,As_test_130)

    dataloader_train = DataLoader(dataset_train, batch_size=Batch_size,shuffle=True)
    dataloader_test_pub = DataLoader(dataset_test_pub, batch_size=Batch_size, shuffle=False)
    dataloader_test_107 = DataLoader(dataset_test_107, batch_size=Batch_size, shuffle=False)
    dataloader_test_130 = DataLoader(dataset_test_130, batch_size=Batch_size, shuffle=False)

    model=Model(vocab_size,embedding_dim,pred_dim_train,dropout,nhead,num_encoder_layers).to(device)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,eta_min=2.0e-5)
    criterion = MCRMSE_NAN_SW_TEW
    metric_func = MCRMSE

    best_loss_pub = np.inf
    best_loss_pri = np.inf
    epochs_without_improvement = 0
    best_model_weights=None

    for epoch in range(epochs):
        model.train(True)
        train_pre_loss=0.0
        #train_mcrmse=0.0
        train_batch_num=0
        for batch_idx,data in enumerate(dataloader_train,0):
            optimizer.zero_grad()

            seqs,strus,loops,labels,As,sam_weig,tew,samaugflag=data

            
            seqs=seqs.to(device)
            strus = strus.to(device)
            loops = loops.to(device)
            labels=labels.to(device)
            As = As.to(device)
            sam_weig=sam_weig.to(device)
            tew=tew.to(device)
            samaugflag=samaugflag.to(device)
            
            pre_out = model(strus,loops,As,seqs)
            
            

            loss_pre = criterion(pre_out[samaugflag==0, :train_seq_scored], labels[samaugflag==0],sam_weig[samaugflag==0],tew[samaugflag==0])+criterion(pre_out[samaugflag==1, (train_seq_len-train_seq_scored):], labels[samaugflag==1],sam_weig[samaugflag==1],tew[samaugflag==1])  #1,+1,because [start]  loss of each site of each sample
            
            loss=loss_pre
            
            

            loss.backward()
            optimizer.step()

            #metric=metric_func(output[:, :seq_scored], labels)

            train_pre_loss+=loss_pre.item()
            
            
            #train_mcrmse+=metric.item()
            train_batch_num=train_batch_num+1
        
        model.train(False)
        test_loss_pub = 0.0
        test_batch_num_pub=0
        for batch_idx, data in enumerate(dataloader_test_pub, 0):
            optimizer.zero_grad()

            seqs,strus,loops,labels,As=data

            
            seqs=seqs.to(device)
            
            strus = strus.to(device)
            loops = loops.to(device)
            labels = labels.to(device)
            As = As.to(device)
            
            pre_out = model(strus,loops,As,seqs)
            
            
            
            loss = metric_func(pre_out[:, :test_seq_scored_pub,:pred_dim_pub], labels)  # 1,+1,because [start]

            #metric = metric_func(output[:, :seq_scored], labels)
            test_loss_pub += loss.item()
            #test_mcrmse+=metric.item()
            test_batch_num_pub=test_batch_num_pub+1

        model.train(False)
        test_loss_107 = 0.0
        test_batch_num_107=0
        for batch_idx, data in enumerate(dataloader_test_107, 0):
            optimizer.zero_grad()

            seqs,strus,loops,labels,As=data

            
            seqs=seqs.to(device)
            

                
                    

            strus = strus.to(device)
            loops = loops.to(device)
            labels = labels.to(device)
            As = As.to(device)
            
            pre_out = model(strus,loops,As,seqs)
            
            
            
            loss = metric_func(pre_out[:, :test_seq_scored_107,:pred_dim_107], labels)  # 1,+1,because [start]

            #metric = metric_func(output[:, :seq_scored], labels)
            test_loss_107 += loss.item()
            #test_mcrmse+=metric.item()
            test_batch_num_107=test_batch_num_107+1

        model.train(False)
        test_loss_130 = 0.0
        test_batch_num_130=0
        for batch_idx, data in enumerate(dataloader_test_130, 0):
            optimizer.zero_grad()

            seqs,strus,loops,labels,As=data

            
            seqs=seqs.to(device)
            
                

            strus = strus.to(device)
            loops = loops.to(device)
            labels = labels.to(device)
            As = As.to(device)
            
            pre_out = model(strus,loops,As,seqs)
            
            
            loss = metric_func(pre_out[:, :test_seq_scored_130,:pred_dim_130], labels)  # 1,+1,because [start]

            #metric = metric_func(output[:, :seq_scored], labels)
            test_loss_130 += loss.item()
            #test_mcrmse+=metric.item()
            test_batch_num_130=test_batch_num_130+1   



        scheduler.step()
        print("epoch: [%d], train loss: [%.6f], public loss: [%.6f], private loss: [%.6f]" % (epoch, train_pre_loss / train_batch_num, test_loss_pub / test_batch_num_pub,((test_loss_107 / test_batch_num_107)+(test_loss_130 / test_batch_num_130))/2.0))
        
        if (test_loss_pub / test_batch_num_pub) < best_loss_pub:
            best_loss_pub = test_loss_pub / test_batch_num_pub
            epochs_without_improvement = 0
            best_model_weights=model.state_dict()
            best_loss_pri=((test_loss_107 / test_batch_num_107)+(test_loss_130 / test_batch_num_130))/2.0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement == patience:
            print('Early stopping at epoch {}'.format(epoch))
            print("best public loss: [%.6f], best private loss: [%.6f]" % (best_loss_pub,best_loss_pri))
            break
        

    torch.save(best_model_weights,"./best_model_weights_withoutpretrain")    
        
train_fold()

