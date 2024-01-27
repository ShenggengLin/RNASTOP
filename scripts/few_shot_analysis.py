# dna pretrain
import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from torch.utils.data import Dataset, DataLoader
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

from Bio import pairwise2
from Bio.Seq import Seq
import pickle

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

seed = 2024
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
learning_rate = 4.0e-4
Batch_size = 64
Conv_kernel = 7
dropout = 0.3
embedding_dim = 128
num_encoder_layers = 4
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
import fm

# Load RNA-FM model
fm_pretrain_model, fm_pretrain_alphabet = fm.pretrained.rna_fm_t12()
fm_pretrain_batch_converter = fm_pretrain_alphabet.get_batch_converter()
fm_pretrain_model = fm_pretrain_model.to(device)

from transformers import AutoTokenizer, AutoModel

dnapt2_PATH = "./DNABERT6/"
dnapt2_tokenizer = AutoTokenizer.from_pretrained(dnapt2_PATH)
dnapt2_model = AutoModel.from_pretrained(dnapt2_PATH).to(device)  # load model

tokens = 'ACGU().BEHIMSXDF'  # D start,F end
vocab_size = len(tokens)

SEED = 4
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

patience = 50
error_alpha = 0.5
error_beta = 5
epochs = 1000
nhead = 4
nStrDim = 8
Use_pretrain_model = True


class RNADataset(Dataset):
    def __init__(self, seqs, seqsOri, Stru, Loop, labels, As, train_sam_wei=None, train_error_weights=None,
                 sam_aug_flag=None):

        self.seqs = seqsOri

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


class RNADataset_nopretrain(Dataset):
    def __init__(self, seqs, seqsOri, Stru, Loop, labels, As, train_sam_wei=None, train_error_weights=None,
                 sam_aug_flag=None):

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


def preprocess_inputs(np_seq):
    re_seq = []
    for i in range(len(np_seq)):
        re_seq.append([tokens.index(s) for s in np_seq[i]])

    re_seq = np.array(re_seq)

    return re_seq


def filter_train(df):
    for i in range(len(df)):
        if df['SN_filter'][i] == 0:

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


def seq_cluster(text_lists, thred=0.75):
    clusters = []
    for i in range(len(text_lists)):
        flag = False
        for cluster in clusters:
            for id in cluster:
                score = pairwise2.align.localxx(Seq(text_lists[i]), Seq(text_lists[id]))[0][2] / max(len(text_lists[i]),
                                                                                                     len(text_lists[
                                                                                                             id]))

                if score >= thred:
                    cluster.add(i)
                    flag = True
                    break
            if flag:
                break
        if not flag:
            clusters.append({i})

    # valid_clusters=[i for i in clusters if len(i)>50]
    num_each_clu = []
    for i in range(len(clusters)):
        num_each_clu.append(len(clusters[i]))
    max_num = max(num_each_clu)
    min_num = min(num_each_clu)

    print('total cluster:{}'.format(len(clusters)))
    print('num of each cluster:{}'.format(clusters))
    print('max_num:{}'.format(max_num))
    print('min_num:{}'.format(min_num))
    return clusters


def blast_similirity(train_seq, test_seq):
    simi_list = []
    for i in range(len(test_seq)):
        temp_list = []
        for j in range(len(train_seq)):
            score = pairwise2.align.localxx(Seq(test_seq[i]), Seq(train_seq[j]))[0][2] / max(len(test_seq[i]),
                                                                                             len(train_seq[j]))
            temp_list.append(score)
        simi_list.append(temp_list)

    return simi_list


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


pred_cols_test = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pred_dim, dropout, nhead, num_encoder_layers):
        super().__init__()

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

        # self.embeddingStru = nn.Embedding(vocab_size, embedding_dim)
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
        embedded_seq_dnapt2_2 = self.conv_encoder_seq_dnapt2(embeddedSeq_dnapt2_share.permute(0, 2, 1)).permute(0, 2, 1)

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


class Model_nopretrain(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pred_dim, dropout, nhead, num_encoder_layers):
        super().__init__()

        self.embeddingSeq = nn.Embedding(vocab_size, embedding_dim)

        # self.embeddingStru = nn.Embedding(vocab_size, embedding_dim)
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

    def forward(self, stru, loop, As, seq):
        # As shape [64, 107, 107, 8]

        embeddedSeq = self.embeddingSeq(seq)

        embeddedLoop = self.embeddingloop(loop)

        As = self.embeddingstr(As.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [64, 107, 1, 128]
        As = torch.squeeze(As)  # [64, 107, 128]

        embeddedSeq_share = self.transformer_encoder_share(embeddedSeq)
        embedded_seq1 = self.transformer_encoder_seq_fm(embeddedSeq_share)
        embedded_seq2 = self.conv_encoder_seq_fm(embeddedSeq_share.permute(0, 2, 1)).permute(0, 2, 1)

        embeddedLoop = self.transformer_encoder_share(embeddedLoop)
        embedded_loop1 = self.transformer_encoder_loop(embeddedLoop)
        embedded_loop2 = self.conv_encoder_loop(embeddedLoop.permute(0, 2, 1)).permute(0, 2, 1)

        As = self.transformer_encoder_share(As)
        embedded_str1 = self.transformer_encoder_str(As)
        embedded_str2 = self.conv_encoder_str(As.permute(0, 2, 1)).permute(0, 2, 1)

        embedded_fus_trans = embedded_seq1 + embedded_loop1 + embedded_str1
        embedded_fus_conv = embedded_seq2 + embedded_loop2 + embedded_str2

        del embedded_loop1, embedded_str1, embedded_loop2, embedded_str2, embedded_seq1, embedded_seq2

        embedded_fus_trans = self.transformer_encoder_fustrans(embedded_fus_trans)
        embedded_fus_conv = self.conv_encoder_fusconv(embedded_fus_conv.permute(0, 2, 1)).permute(0, 2, 1)

        embedded_cat = torch.cat((embedded_fus_trans, embedded_fus_conv), 2)
        embedded_cat = self.conv_encoder_finfus(embedded_cat.permute(0, 2, 1)).permute(0, 2, 1)
        pre_out = self.pre(embedded_cat)

        return pre_out


def mean_squared(y_true, y_pred):
    return torch.mean(torch.sqrt(torch.mean((y_true - y_pred) ** 2)))  # a number =each sample site


def MCRMSE(y_pred, y_true):
    y_true = torch.where(torch.isnan(y_true), y_pred, y_true)

    s = mean_squared(y_true[:, 0], y_pred[:, 0]) / 1.0
    s = s + mean_squared(y_true[:, 1], y_pred[:, 1]) / 1.0
    s = s + mean_squared(y_true[:, 2], y_pred[:, 2]) / 1.0
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


def train_fold():
    train_pubtest_DataFrame = pd.read_json('./data/Kaggle_train.json', lines=True)

    train_pubtest_DataFrame_SN1 = train_pubtest_DataFrame[train_pubtest_DataFrame['SN_filter'] == 1]
    train_pubtest_DataFrame_SN1 = shuffle(train_pubtest_DataFrame_SN1)
    train_pubtest_DataFrame_SN0 = train_pubtest_DataFrame[train_pubtest_DataFrame['SN_filter'] == 0]

    trainDataFrame = pd.concat([train_pubtest_DataFrame_SN1[:-400], train_pubtest_DataFrame_SN0], ignore_index=True)
    trainDataFrame = filter_train(trainDataFrame)  # [:20]

    testDataFrame_130 = pd.read_csv('./data/private_test_labels.csv', encoding='utf-8')
    testDataFrame_130 = testDataFrame_130[testDataFrame_130['test_filter'] == 1]  # [:10]

    print("trainDataFrame shape:", trainDataFrame.shape)
    print("testDataFrame_130 shape:", testDataFrame_130.shape)

    trainSeqOri = trainDataFrame['sequence'].values.tolist()

    testSeqOri_130 = testDataFrame_130['sequence'].values.tolist()

    simi_matrix = blast_similirity(trainSeqOri, testSeqOri_130)

    avg_simi = sum(sum(simi_list) for simi_list in simi_matrix) / (len(simi_matrix) * len(simi_matrix[0]))

    max_simi = 0
    min_simi = 1
    max_simi_id = 0
    min_simi_id = 0
    for i in range(len(simi_matrix)):
        if max(simi_matrix[i]) > max_simi:
            max_simi = max(simi_matrix[i])
            max_simi_id = i
        if min(simi_matrix[i]) < min_simi:
            min_simi = min(simi_matrix[i])
            min_simi_id = i
    print("max_simi_value", max_simi)
    print("min_simi_value", min_simi)
    print("avg_simi_value", avg_simi)

    testSeqOri_130 = np.array([testSeqOri_130[max_simi_id], testSeqOri_130[min_simi_id]])
    testStruOri_130 = np.array([testDataFrame_130['structure'].values.tolist()[max_simi_id],
                                testDataFrame_130['structure'].values.tolist()[min_simi_id]])
    testLoopOri_130 = np.array([testDataFrame_130['predicted_loop_type'].values.tolist()[max_simi_id],
                                testDataFrame_130['predicted_loop_type'].values.tolist()[min_simi_id]])
    testSeq_130 = preprocess_inputs(testSeqOri_130)
    testStru_130 = preprocess_inputs(testStruOri_130)
    testLoop_130 = preprocess_inputs(testLoopOri_130)
    testSeqLen_130 = np.array(testDataFrame_130['seq_length'].values.tolist())
    testStruAdj_130 = get_structure_adj(testSeqLen_130, testStruOri_130, testSeqOri_130)
    test_bp_matrix_130 = []
    for test_seq in testSeqOri_130:
        test_bp_matrix_130.append(bpps(test_seq, package='vienna_2').tolist())
    test_bp_matrix_130 = np.array(test_bp_matrix_130)

    As_test_130 = np.array(test_bp_matrix_130)
    Ds_test_130 = get_distance_matrix(As_test_130)
    DDs_test_130 = get_distance_matrix_2d(testStruAdj_130)
    As_test_130 = np.concatenate([As_test_130[:, :, :, None], testStruAdj_130, Ds_test_130, DDs_test_130],
                                 axis=3).astype(np.float32)

    del testStruAdj_130, Ds_test_130, DDs_test_130

    testData_labels_130 = np.array([testDataFrame_130[pred_cols_test].values.tolist()[max_simi_id],
                                    testDataFrame_130[pred_cols_test].values.tolist()[min_simi_id]])
    testData_labels_num_130 = []
    for i in range(len(testData_labels_130)):
        sam_labels_num = []
        for j in range(len(testData_labels_130[0])):
            temp = testData_labels_130[i][j].lstrip("[").rstrip("]").split(",")
            temp2 = list(filter(lambda x: x != ' None', temp))
            sam_labels_num.append(list(map(float, temp2)))
        testData_labels_num_130.append(sam_labels_num)
    testData_labels_num_130 = np.array(testData_labels_num_130, dtype='float32').transpose((0, 2, 1))
    print("testData_labels_num_130 shape", testData_labels_num_130.shape)
    test_seq_scored_130 = len(testData_labels_num_130[0])
    pred_dim_130 = len(testData_labels_num_130[0][0])

    dataset_test_130 = RNADataset(testSeq_130, testSeqOri_130, testStru_130, testLoop_130, testData_labels_num_130,
                                  As_test_130)
    dataloader_test_130 = DataLoader(dataset_test_130, batch_size=Batch_size, shuffle=False)

    dataset_test_130_nopretrain = RNADataset_nopretrain(testSeq_130, testSeqOri_130, testStru_130, testLoop_130,
                                                        testData_labels_num_130, As_test_130)
    dataloader_test_130_nopretrain = DataLoader(dataset_test_130_nopretrain, batch_size=Batch_size, shuffle=False)

    model = Model(vocab_size, embedding_dim, 5, dropout, nhead, num_encoder_layers).to(device)
    model.load_state_dict(torch.load('./best_model_weights'))
    model.eval()

    model_nopretrain = Model_nopretrain(vocab_size, embedding_dim, 5, dropout, nhead, num_encoder_layers).to(device)
    model_nopretrain.load_state_dict(torch.load('./best_model_weights_withoutpretrain'))
    model_nopretrain.eval()
    metric_func = MCRMSE

    model.train(False)

    for batch_idx, data in enumerate(dataloader_test_130, 0):

        seqs, strus, loops, labels, As = data

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

        pre_out = model(strus, loops, As, fm_seq=fm_seqs, dnapt2_seq=dnapt2_seqs)

        loss1 = metric_func(pre_out[0, :test_seq_scored_130, :pred_dim_130], labels[0, :, :])  # 1,+1,because [start]
        loss2 = metric_func(pre_out[1, :test_seq_scored_130, :pred_dim_130], labels[1, :, :])  # 1,+1,because [start]
        print(loss1.item())
        print(loss2.item())
    for batch_idx, data in enumerate(dataloader_test_130_nopretrain, 0):
        seqs, strus, loops, labels, As = data

        seqs = seqs.to(device)

        strus = strus.to(device)
        loops = loops.to(device)
        labels = labels.to(device)
        As = As.to(device)

        pre_out = model_nopretrain(strus, loops, As, seq=seqs)

        loss1 = metric_func(pre_out[0, :test_seq_scored_130, :pred_dim_130], labels[0, :, :])  # 1,+1,because [start]
        loss2 = metric_func(pre_out[1, :test_seq_scored_130, :pred_dim_130], labels[1, :, :])  # 1,+1,because [start]
        print(loss1.item())
        print(loss2.item())


def train_fold_avg():
    train_pubtest_DataFrame = pd.read_json('./data/Kaggle_train.json', lines=True)

    train_pubtest_DataFrame_SN1 = train_pubtest_DataFrame[train_pubtest_DataFrame['SN_filter'] == 1]
    train_pubtest_DataFrame_SN1 = shuffle(train_pubtest_DataFrame_SN1)
    train_pubtest_DataFrame_SN0 = train_pubtest_DataFrame[train_pubtest_DataFrame['SN_filter'] == 0]

    trainDataFrame = pd.concat([train_pubtest_DataFrame_SN1[:-400], train_pubtest_DataFrame_SN0], ignore_index=True)
    trainDataFrame = filter_train(trainDataFrame)  # [:20]

    testDataFrame_130 = pd.read_csv('./data/private_test_labels.csv', encoding='utf-8')
    testDataFrame_130 = testDataFrame_130[testDataFrame_130['test_filter'] == 1]  # [:10]

    print("trainDataFrame shape:", trainDataFrame.shape)
    print("testDataFrame_130 shape:", testDataFrame_130.shape)

    trainSeqOri = trainDataFrame['sequence'].values.tolist()

    testSeqOri_130 = testDataFrame_130['sequence'].values.tolist()

    simi_matrix = blast_similirity(trainSeqOri, testSeqOri_130)

    avg_simi = sum(sum(simi_list) for simi_list in simi_matrix) / (len(simi_matrix) * len(simi_matrix[0]))

    max_simi = 0
    min_simi = 1
    max_simi_id = 0
    min_simi_id = 0
    for i in range(len(simi_matrix)):
        if max(simi_matrix[i]) > max_simi:
            max_simi = max(simi_matrix[i])
            max_simi_id = i
        if min(simi_matrix[i]) < min_simi:
            min_simi = min(simi_matrix[i])
            min_simi_id = i
    print("max_simi_value", max_simi)
    print("min_simi_value", min_simi)
    print("avg_simi_value", avg_simi)

    testStruOri_130 = np.array(testDataFrame_130['structure'].values.tolist())
    testLoopOri_130 = np.array(testDataFrame_130['predicted_loop_type'].values.tolist())
    testSeq_130 = preprocess_inputs(testSeqOri_130)
    testStru_130 = preprocess_inputs(testStruOri_130)
    testLoop_130 = preprocess_inputs(testLoopOri_130)
    testSeqLen_130 = np.array(testDataFrame_130['seq_length'].values.tolist())
    testStruAdj_130 = get_structure_adj(testSeqLen_130, testStruOri_130, testSeqOri_130)
    test_bp_matrix_130 = []
    for test_seq in testSeqOri_130:
        test_bp_matrix_130.append(bpps(test_seq, package='vienna_2').tolist())
    test_bp_matrix_130 = np.array(test_bp_matrix_130)

    As_test_130 = np.array(test_bp_matrix_130)
    Ds_test_130 = get_distance_matrix(As_test_130)
    DDs_test_130 = get_distance_matrix_2d(testStruAdj_130)
    As_test_130 = np.concatenate([As_test_130[:, :, :, None], testStruAdj_130, Ds_test_130, DDs_test_130],
                                 axis=3).astype(np.float32)

    del testStruAdj_130, Ds_test_130, DDs_test_130

    testData_labels_130 = np.array(testDataFrame_130[pred_cols_test].values.tolist())
    testData_labels_num_130 = []
    for i in range(len(testData_labels_130)):
        sam_labels_num = []
        for j in range(len(testData_labels_130[0])):
            temp = testData_labels_130[i][j].lstrip("[").rstrip("]").split(",")
            temp2 = list(filter(lambda x: x != ' None', temp))
            sam_labels_num.append(list(map(float, temp2)))
        testData_labels_num_130.append(sam_labels_num)
    testData_labels_num_130 = np.array(testData_labels_num_130, dtype='float32').transpose((0, 2, 1))
    print("testData_labels_num_130 shape", testData_labels_num_130.shape)
    test_seq_scored_130 = len(testData_labels_num_130[0])
    pred_dim_130 = len(testData_labels_num_130[0][0])

    dataset_test_130 = RNADataset(testSeq_130, testSeqOri_130, testStru_130, testLoop_130, testData_labels_num_130,
                                  As_test_130)
    dataloader_test_130 = DataLoader(dataset_test_130, batch_size=Batch_size, shuffle=False)

    dataset_test_130_nopretrain = RNADataset_nopretrain(testSeq_130, testSeqOri_130, testStru_130, testLoop_130,
                                                        testData_labels_num_130, As_test_130)
    dataloader_test_130_nopretrain = DataLoader(dataset_test_130_nopretrain, batch_size=Batch_size, shuffle=False)

    model = Model(vocab_size, embedding_dim, 5, dropout, nhead, num_encoder_layers).to(device)
    model.load_state_dict(torch.load('./best_model_weights'))
    model.eval()

    model_nopretrain = Model_nopretrain(vocab_size, embedding_dim, 5, dropout, nhead, num_encoder_layers).to(device)
    model_nopretrain.load_state_dict(torch.load('./best_model_weights_withoutpretrain'))
    model_nopretrain.eval()
    metric_func = MCRMSE

    model.train(False)
    test_loss_130 = 0.0
    test_batch_num_130 = 0

    for batch_idx, data in enumerate(dataloader_test_130, 0):

        seqs, strus, loops, labels, As = data

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

        pre_out = model(strus, loops, As, fm_seq=fm_seqs, dnapt2_seq=dnapt2_seqs)

        loss = metric_func(pre_out[:, :test_seq_scored_130, :pred_dim_130], labels)  # 1,+1,because [start]
        test_loss_130 += loss.item()
        test_batch_num_130 = test_batch_num_130 + 1
    print(test_loss_130 / test_batch_num_130)

    test_loss_130 = 0.0
    test_batch_num_130 = 0
    for batch_idx, data in enumerate(dataloader_test_130_nopretrain, 0):
        seqs, strus, loops, labels, As = data

        seqs = seqs.to(device)

        strus = strus.to(device)
        loops = loops.to(device)
        labels = labels.to(device)
        As = As.to(device)

        pre_out = model_nopretrain(strus, loops, As, seq=seqs)

        loss = metric_func(pre_out[:, :test_seq_scored_130, :pred_dim_130], labels)  # 1,+1,because [start]
        test_loss_130 += loss.item()
        test_batch_num_130 = test_batch_num_130 + 1
    print(test_loss_130 / test_batch_num_130)


tokens_tsne = 'ACGUP'  # D start,F end
vocab_size_tsne = len(tokens_tsne)


def preprocess_inputs_tsne(np_seq):
    re_seq = []
    for i in range(len(np_seq)):
        re_seq.append([tokens_tsne.index(s) for s in np_seq[i]])

    re_seq = np.array(re_seq)

    return re_seq


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    color_list = ["#D7BFA6", "#9CCCCC", "#C7B0C1", "#486090", "#6078A8", "#7890A8", "#B5C9C9", "#90A8C0", "#A8A890"]
    label_list = ['train_107_seqs', 'private_107_seqs', 'private_130_seqs']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(data.shape[0]):
        # plt.text(data[i, 0], data[i, 1], str(label[i]),color=color_list[label[i]],fontdict={'weight': 'bold', 'size': 9})
        # plt.text(data[i, 0], data[i, 1], color=color_list[label[i]])
        plt.scatter(data[i, 0], data[i, 1], c=color_list[label[i]], marker='*')
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)

    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    legend1 = plt.scatter([0], [0], c=color_list[0], marker='*', label=label_list[0])
    legend2 = plt.scatter([0], [0], c=color_list[1], marker='*', label=label_list[1])
    legend3 = plt.scatter([0], [0], c=color_list[2], marker='*', label=label_list[2])
    handles = [legend1, legend2, legend3]

    plt.legend(handles=handles, loc='best', prop={'family': 'Times New Roman', 'size': 15})
    # 设置 x 轴和 y 轴的线条宽度
    plt.gca().spines['bottom'].set_linewidth(1.5)  # 设置 x 轴线条宽度
    plt.gca().spines['left'].set_linewidth(1.5)  # 设置 y 轴线条宽度
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('./few_shot_tsne.png', dpi=500)


def sample_weight(text_lists, thred=20):
    clusters = []
    for text in text_lists:
        flag = False
        for cluster in clusters:
            for _text in cluster:
                if editdistance.distance(text, _text) <= thred:
                    cluster.add(text)
                    flag = True
                    break
            if flag:
                break
        if not flag:
            clusters.append({text})

    # valid_clusters=[i for i in clusters if len(i)>50]
    print('total cluster:{}'.format(len(clusters)))

    clusters_sam_wei = []
    for i in range(len(clusters)):
        clusters_sam_wei.append(1 / np.sqrt(len(clusters[i])))

    sam_wei = []
    for text in text_lists:
        for j in range(len(clusters)):
            if text in clusters[j]:
                sam_wei.append(clusters_sam_wei[j])
                break
    sam_wei = np.array(sam_wei)

    return sam_wei


def tSNE():
    trainDataFrame = pd.read_json('../data/train_dataset.json')
    print('begin cluster...')
    sam_wei = sample_weight(trainDataFrame['sequence'])
    trainDataFrame['sam_wei'] = sam_wei

    testDataFrame_107 = pd.read_csv('../data/test_dataset_private_107.csv', encoding='utf-8')

    testDataFrame_130 = pd.read_csv('../data/test_dataset_private_130.csv', encoding='utf-8')

    print("trainDataFrame shape:", trainDataFrame.shape)
    print("testDataFrame_107 shape:", testDataFrame_107.shape)
    print("testDataFrame_130 shape:", testDataFrame_130.shape)

    trainSeq = trainDataFrame['sequence'].values.tolist()
    for i in range(len(trainSeq)):
        trainSeq[i] = trainSeq[i] + 'P' * 23
    testSeq_107 = testDataFrame_107['sequence'].values.tolist()
    for i in range(len(testSeq_107)):
        testSeq_107[i] = testSeq_107[i] + 'P' * 23
    testSeq_130 = testDataFrame_130['sequence'].values.tolist()

    all_seq = trainSeq + testSeq_107 + testSeq_130

    label = [0] * len(trainSeq) + [1] * len(testSeq_107) + [2] * len(testSeq_130)

    all_seq = preprocess_inputs_tsne(all_seq).tolist()

    embeddingSeq = nn.Embedding(vocab_size_tsne, 16)

    embedding_all_seq = embeddingSeq(torch.Tensor(all_seq).long()).detach().numpy()

    embedding_all_seq_reshape = np.reshape(embedding_all_seq, (
    len(embedding_all_seq), len(embedding_all_seq[0]) * len(embedding_all_seq[0][0])))

    tsne = TSNE(n_components=2, init='pca', method='barnes_hut')
    all_seq_tsne = tsne.fit_transform(np.array(embedding_all_seq_reshape))

    print(all_seq_tsne.shape)
    plot_embedding(all_seq_tsne, label, 't-SNE embedding of the seq')


# train_fold()
# train_fold_avg()
tSNE()
