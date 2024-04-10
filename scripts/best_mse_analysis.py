# Import the required libraries
import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader
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

#The hyperparameters of the model can be modified here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
learning_rate = 4.0e-4
Batch_size = 64
Conv_kernel = 7
dropout = 0.3
embedding_dim = 128
num_encoder_layers = 4
patience = 50
error_alpha = 0.5
error_beta = 5
epochs = 1000
nhead = 4
nStrDim = 8
Use_pretrain_model = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Load RNA-FM model
import fm
fm_pretrain_model, fm_pretrain_alphabet = fm.pretrained.rna_fm_t12()
fm_pretrain_batch_converter = fm_pretrain_alphabet.get_batch_converter()
fm_pretrain_model = fm_pretrain_model.to(device)

# Load DNABERT model
from transformers import AutoTokenizer, AutoModel
dnapt2_PATH = "./DNABERT6/"
dnapt2_tokenizer = AutoTokenizer.from_pretrained(dnapt2_PATH)
dnapt2_model = AutoModel.from_pretrained(dnapt2_PATH).to(device)  # load model

#Definition the word list
tokens = 'ACGU().BEHIMSXDF'  # D start,F end
vocab_size = len(tokens)

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
    def __init__(self, seqs, seqsOri, Stru, Loop, labels, As, train_sam_wei=None, train_error_weights=None,
                 sam_aug_flag=None):
        if Use_pretrain_model:
            self.seqs = seqsOri
        else:
            self.seqs = seqs

        self.Stru = Stru
        self.Loop = Loop
        self.labels = labels
        self.As = As
        self.train_sam_wei = train_sam_wei
        self.train_error_weights = train_error_weights
        self.sam_aug_flag = sam_aug_flag
        self.length = len(self.labels)

    def __getitem__(self, idx):
        if (self.train_sam_wei is not None) and (self.train_error_weights is not None) and (
                self.sam_aug_flag is not None):
            return self.seqs[idx], self.Stru[idx], self.Loop[idx], self.labels[idx], self.As[idx], self.train_sam_wei[
                idx], self.train_error_weights[idx], self.sam_aug_flag[idx]
        else:
            return self.seqs[idx], self.Stru[idx], self.Loop[idx], self.labels[idx], self.As[idx]

    def __len__(self):
        return self.length

#Get the embedding of the mrna sequences
def preprocess_inputs(np_seq):
    re_seq = []
    for i in range(len(np_seq)):
        re_seq.append([tokens.index(s) for s in np_seq[i]])

    re_seq = np.array(re_seq)

    return re_seq

#Get the adjacency matrix of the mrna
def get_structure_adj(data_seq_length, data_structure, data_sequence):
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

    Ss = np.array(Ss, dtype='float32')

    '''new = np.zeros((Ss.shape[0], Ss.shape[1] + 2, Ss.shape[2] + 2, Ss.shape[3]))
    new[:, 1:-1, 1:-1, :] = Ss'''

    return Ss

# Define the RNADegpre model
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pred_dim, dropout, nhead, num_encoder_layers):
        super().__init__()

        if Use_pretrain_model:
            self.embeddingSeq_fm = nn.Sequential(nn.Linear(640, 320),
                                                 nn.LayerNorm(320),
                                                 nn.ReLU(),
                                                 nn.Dropout(dropout),
                                                 nn.Linear(320, embedding_dim)
                                                 )
            self.embeddingSeq_dnapt2 = nn.Sequential(nn.Linear(768, 256),
                                                     nn.LayerNorm(256),
                                                     nn.ReLU(),
                                                     nn.Dropout(dropout),
                                                     nn.Linear(256, embedding_dim)
                                                     )
        else:
            self.embeddingSeq = nn.Embedding(vocab_size, embedding_dim)

        self.embeddingloop = nn.Embedding(vocab_size, embedding_dim)

        self.embeddingstr = nn.Sequential(nn.Conv2d(nStrDim, embedding_dim, 1),
                                          nn.BatchNorm2d(embedding_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(dropout),
                                          nn.AdaptiveAvgPool2d((None, 1))
                                          )

        encoder_layer_share = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout,
                                                         batch_first=True)
        self.transformer_encoder_share = nn.TransformerEncoder(encoder_layer_share, num_layers=num_encoder_layers)

        encoder_layer_seq_fm = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout,
                                                          batch_first=True)
        self.transformer_encoder_seq_fm = nn.TransformerEncoder(encoder_layer_seq_fm, num_layers=num_encoder_layers)
        self.conv_encoder_seq_fm = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim, Conv_kernel),
                                                 nn.BatchNorm1d(num_features=embedding_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(dropout),

                                                 nn.ConvTranspose1d(embedding_dim, embedding_dim, Conv_kernel),
                                                 nn.BatchNorm1d(num_features=embedding_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(dropout),
                                                 )
        encoder_layer_seq_dnapt2 = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout,
                                                              batch_first=True)
        self.transformer_encoder_seq_dnapt2 = nn.TransformerEncoder(encoder_layer_seq_dnapt2,
                                                                    num_layers=num_encoder_layers)
        self.conv_encoder_seq_dnapt2 = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim, Conv_kernel),
                                                     nn.BatchNorm1d(num_features=embedding_dim),
                                                     nn.ReLU(inplace=True),
                                                     nn.Dropout(dropout),

                                                     nn.ConvTranspose1d(embedding_dim, embedding_dim, Conv_kernel),
                                                     nn.BatchNorm1d(num_features=embedding_dim),
                                                     nn.ReLU(inplace=True),
                                                     nn.Dropout(dropout),
                                                     )

        encoder_layer_loop = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder_loop = nn.TransformerEncoder(encoder_layer_loop, num_layers=num_encoder_layers)
        self.conv_encoder_loop = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim, Conv_kernel),
                                               nn.BatchNorm1d(num_features=embedding_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(dropout),

                                               nn.ConvTranspose1d(embedding_dim, embedding_dim, Conv_kernel),
                                               nn.BatchNorm1d(num_features=embedding_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(dropout),
                                               )

        encoder_layer_str = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout,
                                                       batch_first=True)
        self.transformer_encoder_str = nn.TransformerEncoder(encoder_layer_str, num_layers=num_encoder_layers)
        self.conv_encoder_str = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim, Conv_kernel),
                                              nn.BatchNorm1d(num_features=embedding_dim),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(dropout),

                                              nn.ConvTranspose1d(embedding_dim, embedding_dim, Conv_kernel),
                                              nn.BatchNorm1d(num_features=embedding_dim),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(dropout),
                                              )

        encoder_layer_fustrans = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout,
                                                            batch_first=True)
        self.transformer_encoder_fustrans = nn.TransformerEncoder(encoder_layer_fustrans, num_layers=num_encoder_layers)
        self.conv_encoder_fusconv = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim * 2, Conv_kernel),
                                                  nn.BatchNorm1d(num_features=embedding_dim * 2),
                                                  nn.ReLU(inplace=True),
                                                  nn.Dropout(dropout),

                                                  nn.ConvTranspose1d(embedding_dim * 2, embedding_dim, Conv_kernel),
                                                  nn.BatchNorm1d(num_features=embedding_dim),
                                                  nn.ReLU(inplace=True),
                                                  nn.Dropout(dropout),
                                                  )

        self.conv_encoder_finfus = nn.Sequential(nn.Conv1d(embedding_dim * 2, embedding_dim * 4, Conv_kernel),
                                                 nn.BatchNorm1d(num_features=embedding_dim * 4),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(dropout),

                                                 nn.ConvTranspose1d(embedding_dim * 4, embedding_dim * 2, Conv_kernel),
                                                 nn.BatchNorm1d(num_features=embedding_dim * 2),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(dropout),
                                                 )
        self.pre = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim),
                                 nn.LayerNorm(embedding_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(embedding_dim, pred_dim)
                                 )

    def forward(self, stru, loop, As, seq=None, fm_seq=None, dnapt2_seq=None):
        # As shape [64, 107, 107, 8]

        if not Use_pretrain_model:
            embeddedSeq = self.embeddingSeq(seq)

            embeddedLoop = self.embeddingloop(loop)

            As = self.embeddingstr(As.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [64, 107, 1, 128]
            As = torch.squeeze(As)  # [64, 107, 128]

            embeddedSeq = self.encoder_share(embeddedSeq)
            embedded_seq1 = self.transformer_encoder_seq(embeddedSeq)
            embedded_seq2 = self.conv_encoder_seq(embeddedSeq.permute(0, 2, 1)).permute(0, 2, 1)

            embeddedLoop = self.encoder_share(embeddedLoop)
            embedded_loop1 = self.transformer_encoder_loop(embeddedLoop)
            embedded_loop2 = self.conv_encoder_loop(embeddedLoop.permute(0, 2, 1)).permute(0, 2, 1)

            As = self.encoder_share(As)
            embedded_str1 = self.transformer_encoder_str(As)
            embedded_str2 = self.conv_encoder_str(As.permute(0, 2, 1)).permute(0, 2, 1)

            embedded_fus_trans = embedded_seq1 + embedded_loop1 + embedded_str1
            embedded_fus_conv = embedded_seq2 + embedded_loop2 + embedded_str2

            del embedded_seq1, embedded_loop1, embedded_str1, embedded_seq2, embedded_loop2, embedded_str2

            embedded_fus_trans = self.transformer_encoder_fustrans(embedded_fus_trans)
            embedded_fus_conv = self.conv_encoder_fusconv(embedded_fus_conv.permute(0, 2, 1)).permute(0, 2, 1)

            embedded_cat = torch.cat((embedded_fus_trans, embedded_fus_conv), 2)
            pre_out = self.pre(embedded_cat)

            return pre_out
        else:
            embeddedSeq_fm = self.embeddingSeq_fm(fm_seq)
            embeddedSeq_dnapt2 = self.embeddingSeq_dnapt2(dnapt2_seq)

            embeddedLoop = self.embeddingloop(loop)

            As = self.embeddingstr(As.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [64, 107, 1, 128]
            As = torch.squeeze(As)  # [64, 107, 128]

            embeddedSeq_fm_share = self.transformer_encoder_share(embeddedSeq_fm)
            embedded_seq_fm1 = self.transformer_encoder_seq_fm(embeddedSeq_fm_share)
            embedded_seq_fm2 = self.conv_encoder_seq_fm(embeddedSeq_fm_share.permute(0, 2, 1)).permute(0, 2, 1)

            embeddedSeq_dnapt2_share = self.transformer_encoder_share(embeddedSeq_dnapt2)
            embedded_seq_dnapt2_1 = self.transformer_encoder_seq_dnapt2(embeddedSeq_dnapt2_share)
            embedded_seq_dnapt2_2 = self.conv_encoder_seq_dnapt2(embeddedSeq_dnapt2_share.permute(0, 2, 1)).permute(0,
                                                                                                                    2,
                                                                                                                    1)

            embeddedLoop = self.transformer_encoder_share(embeddedLoop)
            embedded_loop1 = self.transformer_encoder_loop(embeddedLoop)
            embedded_loop2 = self.conv_encoder_loop(embeddedLoop.permute(0, 2, 1)).permute(0, 2, 1)

            As = self.transformer_encoder_share(As)
            embedded_str1 = self.transformer_encoder_str(As)
            embedded_str2 = self.conv_encoder_str(As.permute(0, 2, 1)).permute(0, 2, 1)

            embedded_fus_trans = embedded_seq_fm1 + embedded_loop1 + embedded_str1 + embedded_seq_dnapt2_1
            embedded_fus_conv = embedded_seq_fm2 + embedded_loop2 + embedded_str2 + embedded_seq_dnapt2_2

            del embedded_loop1, embedded_str1, embedded_loop2, embedded_str2, embedded_seq_fm1, embedded_seq_fm2, embedded_seq_dnapt2_1, embedded_seq_dnapt2_2

            embedded_fus_trans = self.transformer_encoder_fustrans(embedded_fus_trans)
            embedded_fus_conv = self.conv_encoder_fusconv(embedded_fus_conv.permute(0, 2, 1)).permute(0, 2, 1)

            embedded_cat = torch.cat((embedded_fus_trans, embedded_fus_conv), 2)
            embedded_cat = self.conv_encoder_finfus(embedded_cat.permute(0, 2, 1)).permute(0, 2, 1)
            pre_out = self.pre(embedded_cat)

            return pre_out

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
    lst_x, lst_y = np.where(d == n)
    for c, x in enumerate(lst_x):
        y = lst_y[c]
        if x + 1 < dim:
            d[x + 1, y] = min(d[x + 1, y], n + 1)
        if y + 1 < dim:
            d[x, y + 1] = min(d[x, y + 1], n + 1)
        if x - 1 >= 0:
            d[x - 1, y] = min(d[x - 1, y], n + 1)
        if y - 1 >= 0:
            d[x, y - 1] = min(d[x, y - 1], n + 1)
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

# Defined loss function
def mean_squared(y_true, y_pred):
    return torch.mean(torch.sqrt(torch.mean((y_true - y_pred) ** 2, axis=1)))  # a number =each sample site
def MCRMSE(y_pred, y_true):
    y_true = torch.where(torch.isnan(y_true), y_pred, y_true)

    s = mean_squared(y_true[:, :, 0], y_pred[:, :, 0]) / 1.0
    s = s + mean_squared(y_true[:, :, 1], y_pred[:, :, 1]) / 1.0
    s = s + mean_squared(y_true[:, :, 2], y_pred[:, :, 2]) / 1.0
    s = s / 3.0
    return s
def mean_squared_MSE(y_true, y_pred):
    return torch.mean(torch.sqrt(torch.mean((y_true - y_pred) ** 2)))  # a number =each sample site
def MSE(y_pred, y_true):
    y_true = torch.where(torch.isnan(y_true), y_pred, y_true)
    s = []
    for i in range(len(y_pred)):
        s.append((mean_squared_MSE(y_true[i], y_pred[i]) / 1.0).item())

    return s

# Define different data types
pred_cols_test = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

#Standardize the list data
def nor_list(list_a):
    max_val = max(list_a)
    min_val = min(list_a)
    nor_list_a = [(x - min_val) / (max_val - min_val) for x in list_a]
    return nor_list_a

#The mRNA sequence is sorted according to the MSE of the mRNA sequence, and the sequenced mRNA is output
def test():
    testDataFrame_107 = pd.read_csv('../data/test_dataset_private_107.csv', encoding='utf-8')

    testDataFrame_130 = pd.read_csv('../data/test_dataset_private_130.csv', encoding='utf-8')

    print("testDataFrame_107 shape:", testDataFrame_107.shape)
    print("testDataFrame_130 shape:", testDataFrame_130.shape)

    testSeqOri_107 = np.array(testDataFrame_107['sequence'].values.tolist())
    testSeqOri_130 = np.array(testDataFrame_130['sequence'].values.tolist())

    testStruOri_107 = np.array(testDataFrame_107['structure'].values.tolist())
    testStruOri_130 = np.array(testDataFrame_130['structure'].values.tolist())

    testLoopOri_107 = np.array(testDataFrame_107['predicted_loop_type'].values.tolist())
    testLoopOri_130 = np.array(testDataFrame_130['predicted_loop_type'].values.tolist())

    testSeq_107 = preprocess_inputs(testSeqOri_107)
    testSeq_130 = preprocess_inputs(testSeqOri_130)

    testStru_107 = preprocess_inputs(testStruOri_107)
    testStru_130 = preprocess_inputs(testStruOri_130)

    testLoop_107 = preprocess_inputs(testLoopOri_107)
    testLoop_130 = preprocess_inputs(testLoopOri_130)

    print('testSeq_107 shape:', testSeq_107.shape)
    print('testSeq_130 shape:', testSeq_130.shape)

    testSeqLen_107 = np.array(testDataFrame_107['seq_length'].values.tolist())
    testSeqLen_130 = np.array(testDataFrame_130['seq_length'].values.tolist())

    testStruAdj_107 = get_structure_adj(testSeqLen_107, testStruOri_107, testSeqOri_107)
    testStruAdj_130 = get_structure_adj(testSeqLen_130, testStruOri_130, testSeqOri_130)

    test_bp_matrix_107 = []
    for test_seq in testSeqOri_107:
        test_bp_matrix_107.append(bpps(test_seq, package='vienna_2').tolist())
    test_bp_matrix_107 = np.array(test_bp_matrix_107)

    test_bp_matrix_130 = []
    for test_seq in testSeqOri_130:
        test_bp_matrix_130.append(bpps(test_seq, package='vienna_2').tolist())
    test_bp_matrix_130 = np.array(test_bp_matrix_130)

    As_test_107 = np.array(test_bp_matrix_107)
    As_test_130 = np.array(test_bp_matrix_130)

    Ds_test_107 = get_distance_matrix(As_test_107)
    Ds_test_130 = get_distance_matrix(As_test_130)

    DDs_test_107 = get_distance_matrix_2d(testStruAdj_107)
    DDs_test_130 = get_distance_matrix_2d(testStruAdj_130)

    As_test_107 = np.concatenate([As_test_107[:, :, :, None], testStruAdj_107, Ds_test_107, DDs_test_107],
                                 axis=3).astype(np.float32)
    As_test_130 = np.concatenate([As_test_130[:, :, :, None], testStruAdj_130, Ds_test_130, DDs_test_130],
                                 axis=3).astype(np.float32)

    del testStruAdj_107, Ds_test_107, DDs_test_107, testStruAdj_130, Ds_test_130, DDs_test_130

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
    testData_labels_num_107 = np.array(testData_labels_num_107, dtype='float32').transpose((0, 2, 1))

    testData_labels_num_130 = []
    for i in range(len(testData_labels_130)):
        sam_labels_num = []
        for j in range(len(testData_labels_130[0])):
            temp = testData_labels_130[i][j].lstrip("[").rstrip("]").split(",")
            temp2 = list(filter(lambda x: x != ' None', temp))
            sam_labels_num.append(list(map(float, temp2)))
        testData_labels_num_130.append(sam_labels_num)
    testData_labels_num_130 = np.array(testData_labels_num_130, dtype='float32').transpose((0, 2, 1))

    print("testData_labels_num_107 shape", testData_labels_num_107.shape)
    print("testData_labels_num_130 shape", testData_labels_num_130.shape)

    test_seq_scored_107 = len(testData_labels_num_107[0])
    pred_dim_107 = len(testData_labels_num_107[0][0])

    test_seq_scored_130 = len(testData_labels_num_130[0])
    pred_dim_130 = len(testData_labels_num_130[0][0])

    dataset_test_107 = RNADataset(testSeq_107, testSeqOri_107, testStru_107, testLoop_107, testData_labels_num_107,
                                  As_test_107)
    dataset_test_130 = RNADataset(testSeq_130, testSeqOri_130, testStru_130, testLoop_130, testData_labels_num_130,
                                  As_test_130)

    dataloader_test_107 = DataLoader(dataset_test_107, batch_size=Batch_size, shuffle=False)
    dataloader_test_130 = DataLoader(dataset_test_130, batch_size=Batch_size, shuffle=False)

    model = Model(vocab_size, embedding_dim, 5, dropout, nhead, num_encoder_layers).to(device)
    model.load_state_dict(torch.load('./best_model_weights'))
    model.eval()
    metric_func = MCRMSE
    metric_func_single = MSE

    model.train(False)
    test_loss_107 = 0.0
    test_loss_107_rea = []
    test_loss_107_ph10 = []
    test_loss_107_50c = []
    model_predict_107_rea = []
    model_predict_107_ph10 = []
    model_predict_107_50c = []
    label_107_rea = []
    label_107_ph10 = []
    label_107_50c = []
    test_batch_num_107 = 0
    for batch_idx, data in enumerate(dataloader_test_107, 0):

        seqs, strus, loops, labels, As = data

        if not Use_pretrain_model:
            seqs = seqs.to(device)
        else:
            fm_pretrain_model.eval()
            seqs_pretrainformat_fm = []
            seqs_dnapt2_id = []

            for temp_i in range(len(seqs)):
                seqs_pretrainformat_fm.append(("RNA", seqs[temp_i]))
                temp_dnapt2_id = dnapt2_tokenizer(seqs[temp_i], return_tensors='pt')["input_ids"].tolist()[0]
                temp_dnapt2_id = temp_dnapt2_id + [3] * (len(seqs[0]) - len(temp_dnapt2_id))

                seqs_dnapt2_id.append(temp_dnapt2_id)

            _, _, fm_pre_batch_tokens = fm_pretrain_batch_converter(seqs_pretrainformat_fm)
            with torch.no_grad():
                fm_pre_results = fm_pretrain_model(fm_pre_batch_tokens.to(device), repr_layers=[12])
            fm_seqs = fm_pre_results["representations"][12][:, 1:-1, :]

            dnapt2_model.eval()
            with torch.no_grad():

                seqs_dnapt2_id = torch.as_tensor(np.array(seqs_dnapt2_id))
                dnapt2_seqs = dnapt2_model(seqs_dnapt2_id.to(device))[0]  # B,S,768

        strus = strus.to(device)
        loops = loops.to(device)
        labels = labels.to(device)
        As = As.to(device)
        if not Use_pretrain_model:
            pre_out = model(strus, loops, As, seq=seqs)
        else:
            pre_out = model(strus, loops, As, fm_seq=fm_seqs, dnapt2_seq=dnapt2_seqs)

        loss = metric_func(pre_out[:, :test_seq_scored_107, :pred_dim_107], labels)  # 1,+1,because [start]
        test_loss_107_rea.extend(
            metric_func_single(pre_out[:, :test_seq_scored_107, 0], labels[:, :, 0]))  # 1,+1,because [start]
        test_loss_107_ph10.extend(
            metric_func_single(pre_out[:, :test_seq_scored_107, 1], labels[:, :, 1]))  # 1,+1,because [start]
        test_loss_107_50c.extend(
            metric_func_single(pre_out[:, :test_seq_scored_107, 2], labels[:, :, 2]))  # 1,+1,because [start]
        model_predict_107_rea.extend(pre_out[:, :test_seq_scored_107, 0].cpu().tolist())
        model_predict_107_ph10.extend(pre_out[:, :test_seq_scored_107, 1].cpu().tolist())
        model_predict_107_50c.extend(pre_out[:, :test_seq_scored_107, 2].cpu().tolist())
        label_107_rea.extend(labels[:, :, 0].cpu().tolist())
        label_107_ph10.extend(labels[:, :, 1].cpu().tolist())
        label_107_50c.extend(labels[:, :, 2].cpu().tolist())

        test_loss_107 += loss.item()

        test_batch_num_107 = test_batch_num_107 + 1

    model.train(False)
    test_loss_130 = 0.0
    test_loss_130_rea = []
    test_loss_130_ph10 = []
    test_loss_130_50c = []
    model_predict_130_rea = []
    model_predict_130_ph10 = []
    model_predict_130_50c = []
    label_130_rea = []
    label_130_ph10 = []
    label_130_50c = []
    test_batch_num_130 = 0
    for batch_idx, data in enumerate(dataloader_test_130, 0):

        seqs, strus, loops, labels, As = data

        if not Use_pretrain_model:
            seqs = seqs.to(device)
        else:
            fm_pretrain_model.eval()
            seqs_pretrainformat_fm = []
            seqs_dnapt2_id = []

            for temp_i in range(len(seqs)):
                seqs_pretrainformat_fm.append(("RNA", seqs[temp_i]))
                temp_dnapt2_id = dnapt2_tokenizer(seqs[temp_i], return_tensors='pt')["input_ids"].tolist()[0]
                temp_dnapt2_id = temp_dnapt2_id + [3] * (len(seqs[0]) - len(temp_dnapt2_id))

                seqs_dnapt2_id.append(temp_dnapt2_id)

            _, _, fm_pre_batch_tokens = fm_pretrain_batch_converter(seqs_pretrainformat_fm)
            with torch.no_grad():
                fm_pre_results = fm_pretrain_model(fm_pre_batch_tokens.to(device), repr_layers=[12])
            fm_seqs = fm_pre_results["representations"][12][:, 1:-1, :]

            dnapt2_model.eval()
            with torch.no_grad():

                seqs_dnapt2_id = torch.as_tensor(np.array(seqs_dnapt2_id))
                dnapt2_seqs = dnapt2_model(seqs_dnapt2_id.to(device))[0]  # B,S,768

        strus = strus.to(device)
        loops = loops.to(device)
        labels = labels.to(device)
        As = As.to(device)
        if not Use_pretrain_model:
            pre_out = model(strus, loops, As, seq=seqs)
        else:
            pre_out = model(strus, loops, As, fm_seq=fm_seqs, dnapt2_seq=dnapt2_seqs)

        loss = metric_func(pre_out[:, :test_seq_scored_130, :pred_dim_130], labels)  # 1,+1,because [start]
        test_loss_130_rea.extend(
            metric_func_single(pre_out[:, :test_seq_scored_130, 0], labels[:, :, 0]))  # 1,+1,because [start]
        test_loss_130_ph10.extend(
            metric_func_single(pre_out[:, :test_seq_scored_130, 1], labels[:, :, 1]))  # 1,+1,because [start]
        test_loss_130_50c.extend(
            metric_func_single(pre_out[:, :test_seq_scored_130, 2], labels[:, :, 2]))  # 1,+1,because [start]
        model_predict_130_rea.extend(pre_out[:, :test_seq_scored_130, 0].cpu().tolist())
        model_predict_130_ph10.extend(pre_out[:, :test_seq_scored_130, 1].cpu().tolist())
        model_predict_130_50c.extend(pre_out[:, :test_seq_scored_130, 2].cpu().tolist())
        label_130_rea.extend(labels[:, :, 0].cpu().tolist())
        label_130_ph10.extend(labels[:, :, 1].cpu().tolist())
        label_130_50c.extend(labels[:, :, 2].cpu().tolist())

        # metric = metric_func(output[:, :seq_scored], labels)
        test_loss_130 += loss.item()
        # test_mcrmse+=metric.item()
        test_batch_num_130 = test_batch_num_130 + 1

    del testStruOri_107, testStruOri_130, testLoopOri_107, testLoopOri_130, testSeq_107, testSeq_130, testStru_107, testStru_130

    testLoop_107, testLoop_130, testSeqLen_107, testSeqLen_130, testData_labels_num_107, testData_labels_num_130
    sorted_id_107_rea = sorted(range(len(test_loss_107_rea)), key=lambda k: test_loss_107_rea[k])
    sorted_id_107_rea_seq = testSeqOri_107[sorted_id_107_rea].tolist()
    sorted_model_predict_107_rea = np.array(model_predict_107_rea)[sorted_id_107_rea].tolist()
    sorted_model_predict_107_rea_nor = []
    for i in range(len(sorted_model_predict_107_rea)):
        sorted_model_predict_107_rea_nor.append(nor_list(sorted_model_predict_107_rea[i]))
    del model_predict_107_rea
    sorted_label_107_rea = np.array(label_107_rea)[sorted_id_107_rea].tolist()
    sorted_label_107_rea_nor = []
    for i in range(len(sorted_label_107_rea)):
        sorted_label_107_rea_nor.append(nor_list(sorted_label_107_rea[i]))
    del label_107_rea
    sorted_id_107_ph10 = sorted(range(len(test_loss_107_ph10)), key=lambda k: test_loss_107_ph10[k])
    sorted_id_107_ph10_seq = testSeqOri_107[sorted_id_107_ph10].tolist()
    sorted_model_predict_107_ph10 = np.array(model_predict_107_ph10)[sorted_id_107_ph10].tolist()
    sorted_model_predict_107_ph10_nor = []
    for i in range(len(sorted_model_predict_107_ph10)):
        sorted_model_predict_107_ph10_nor.append(nor_list(sorted_model_predict_107_ph10[i]))
    del model_predict_107_ph10
    sorted_label_107_ph10 = np.array(label_107_ph10)[sorted_id_107_ph10].tolist()
    sorted_label_107_ph10_nor = []
    for i in range(len(sorted_label_107_ph10)):
        sorted_label_107_ph10_nor.append(nor_list(sorted_label_107_ph10[i]))
    del label_107_ph10
    sorted_id_107_50c = sorted(range(len(test_loss_107_50c)), key=lambda k: test_loss_107_50c[k])
    sorted_id_107_50c_seq = testSeqOri_107[sorted_id_107_50c].tolist()
    sorted_model_predict_107_50c = np.array(model_predict_107_50c)[sorted_id_107_50c].tolist()
    sorted_model_predict_107_50c_nor = []
    for i in range(len(sorted_model_predict_107_50c)):
        sorted_model_predict_107_50c_nor.append(nor_list(sorted_model_predict_107_50c[i]))
    del model_predict_107_50c
    sorted_label_107_50c = np.array(label_107_50c)[sorted_id_107_50c].tolist()
    sorted_label_107_50c_nor = []
    for i in range(len(sorted_label_107_50c)):
        sorted_label_107_50c_nor.append(nor_list(sorted_label_107_50c[i]))
    del label_107_50c
    sorted_id_130_rea = sorted(range(len(test_loss_130_rea)), key=lambda k: test_loss_130_rea[k])
    sorted_id_130_rea_seq = testSeqOri_130[sorted_id_130_rea].tolist()
    sorted_model_predict_130_rea = np.array(model_predict_130_rea)[sorted_id_130_rea].tolist()
    sorted_model_predict_130_rea_nor = []
    for i in range(len(sorted_model_predict_130_rea)):
        sorted_model_predict_130_rea_nor.append(nor_list(sorted_model_predict_130_rea[i]))
    del model_predict_130_rea
    sorted_label_130_rea = np.array(label_130_rea)[sorted_id_130_rea].tolist()
    sorted_label_130_rea_nor = []
    for i in range(len(sorted_label_130_rea)):
        sorted_label_130_rea_nor.append(nor_list(sorted_label_130_rea[i]))
    del label_130_rea
    sorted_id_130_ph10 = sorted(range(len(test_loss_130_ph10)), key=lambda k: test_loss_130_ph10[k])
    sorted_id_130_ph10_seq = testSeqOri_130[sorted_id_130_ph10].tolist()
    sorted_model_predict_130_ph10 = np.array(model_predict_130_ph10)[sorted_id_130_ph10].tolist()
    sorted_model_predict_130_ph10_nor = []
    for i in range(len(sorted_model_predict_130_ph10)):
        sorted_model_predict_130_ph10_nor.append(nor_list(sorted_model_predict_130_ph10[i]))
    del model_predict_130_ph10
    sorted_label_130_ph10 = np.array(label_130_ph10)[sorted_id_130_ph10].tolist()
    sorted_label_130_ph10_nor = []
    for i in range(len(sorted_label_130_ph10)):
        sorted_label_130_ph10_nor.append(nor_list(sorted_label_130_ph10[i]))
    del label_130_ph10
    sorted_id_130_50c = sorted(range(len(test_loss_130_50c)), key=lambda k: test_loss_130_50c[k])
    sorted_id_130_50c_seq = testSeqOri_130[sorted_id_130_50c].tolist()
    sorted_model_predict_130_50c = np.array(model_predict_130_50c)[sorted_id_130_50c].tolist()
    sorted_model_predict_130_50c_nor = []
    for i in range(len(sorted_model_predict_130_50c)):
        sorted_model_predict_130_50c_nor.append(nor_list(sorted_model_predict_130_50c[i]))
    del model_predict_130_50c
    sorted_label_130_50c = np.array(label_130_50c)[sorted_id_130_50c].tolist()
    sorted_label_130_50c_nor = []
    for i in range(len(sorted_label_130_50c)):
        sorted_label_130_50c_nor.append(nor_list(sorted_label_130_50c[i]))
    del label_130_50c

    test_loss_107_rea.sort()
    test_loss_107_ph10.sort()
    test_loss_107_50c.sort()
    test_loss_130_rea.sort()
    test_loss_130_ph10.sort()
    test_loss_130_50c.sort()

    name_107 = ['test_loss_107_rea', 'sorted_id_107_rea_seq', 'predict_107_rea', 'predict_107_rea_nor',
                'sorted_label_107_rea', 'sorted_label_107_rea_nor', 'test_loss_107_ph10', 'sorted_id_107_ph10_seq',
                'predict_107_ph10', 'predict_107_ph10_nor', 'sorted_label_107_ph10', 'sorted_label_107_ph10_nor',
                'test_loss_107_50c', 'sorted_id_107_50c_seq', 'predict_107_50c', 'predict_107_50c_nor',
                'sorted_label_107_50c', 'sorted_label_107_50c_nor']
    name_130 = ['test_loss_130_rea', 'sorted_id_130_rea_seq', 'predict_130_rea', 'predict_130_rea_nor',
                'sorted_label_130_rea', 'sorted_label_130_rea_nor', 'test_loss_130_ph10', 'sorted_id_130_ph10_seq',
                'predict_130_ph10', 'predict_130_ph10_nor', 'sorted_label_130_ph10', 'sorted_label_130_ph10_nor',
                'test_loss_130_50c', 'sorted_id_130_50c_seq', 'predict_130_50c', 'predict_130_50c_nor',
                'sorted_label_130_50c', 'sorted_label_130_50c_nor']
    list_data_107 = []
    list_data_130 = []
    list_data_107.append(test_loss_107_rea)
    del test_loss_107_rea
    list_data_107.append(sorted_id_107_rea_seq)
    del sorted_id_107_rea_seq
    list_data_107.append(sorted_model_predict_107_rea)
    del sorted_model_predict_107_rea
    list_data_107.append(sorted_model_predict_107_rea_nor)
    del sorted_model_predict_107_rea_nor
    list_data_107.append(sorted_label_107_rea)
    del sorted_label_107_rea
    list_data_107.append(sorted_label_107_rea_nor)
    del sorted_label_107_rea_nor
    list_data_130.append(test_loss_130_rea)
    del test_loss_130_rea
    list_data_130.append(sorted_id_130_rea_seq)
    del sorted_id_130_rea_seq
    list_data_130.append(sorted_model_predict_130_rea)
    del sorted_model_predict_130_rea
    list_data_130.append(sorted_model_predict_130_rea_nor)
    del sorted_model_predict_130_rea_nor
    list_data_130.append(sorted_label_130_rea)
    del sorted_label_130_rea
    list_data_130.append(sorted_label_130_rea_nor)
    del sorted_label_130_rea_nor
    list_data_107.append(test_loss_107_ph10)
    del test_loss_107_ph10
    list_data_107.append(sorted_id_107_ph10_seq)
    del sorted_id_107_ph10_seq
    list_data_107.append(sorted_model_predict_107_ph10)
    del sorted_model_predict_107_ph10
    list_data_107.append(sorted_model_predict_107_ph10_nor)
    del sorted_model_predict_107_ph10_nor
    list_data_107.append(sorted_label_107_ph10)
    del sorted_label_107_ph10
    list_data_107.append(sorted_label_107_ph10_nor)
    del sorted_label_107_ph10_nor
    list_data_130.append(test_loss_130_ph10)
    del test_loss_130_ph10
    list_data_130.append(sorted_id_130_ph10_seq)
    del sorted_id_130_ph10_seq
    list_data_130.append(sorted_model_predict_130_ph10)
    del sorted_model_predict_130_ph10
    list_data_130.append(sorted_model_predict_130_ph10_nor)
    del sorted_model_predict_130_ph10_nor
    list_data_130.append(sorted_label_130_ph10)
    del sorted_label_130_ph10
    list_data_130.append(sorted_label_130_ph10_nor)
    del sorted_label_130_ph10_nor
    list_data_107.append(test_loss_107_50c)
    del test_loss_107_50c
    list_data_107.append(sorted_id_107_50c_seq)
    del sorted_id_107_50c_seq
    list_data_107.append(sorted_model_predict_107_50c)
    del sorted_model_predict_107_50c
    list_data_107.append(sorted_model_predict_107_50c_nor)
    del sorted_model_predict_107_50c_nor
    list_data_107.append(sorted_label_107_50c)
    del sorted_label_107_50c
    list_data_107.append(sorted_label_107_50c_nor)
    del sorted_label_107_50c_nor
    list_data_130.append(test_loss_130_50c)
    del test_loss_130_50c
    list_data_130.append(sorted_id_130_50c_seq)
    del sorted_id_130_50c_seq
    list_data_130.append(sorted_model_predict_130_50c)
    del sorted_model_predict_130_50c
    list_data_130.append(sorted_model_predict_130_50c_nor)
    del sorted_model_predict_130_50c_nor
    list_data_130.append(sorted_label_130_50c)
    del sorted_label_130_50c
    list_data_130.append(sorted_label_130_50c_nor)
    del sorted_label_130_50c_nor
    sort_output_107 = pd.DataFrame(dict(zip(name_107, list_data_107)))
    sort_output_107.to_csv('./sort_output_107.csv')

    sort_output_130 = pd.DataFrame(dict(zip(name_130, list_data_130)))
    sort_output_130.to_csv('./sort_output_130.csv')

    print(
        "private loss: [%.6f]" % (((test_loss_107 / test_batch_num_107) + (test_loss_130 / test_batch_num_130)) / 2.0))


test()
