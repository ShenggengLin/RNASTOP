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

def main():
    
    testDataFrame=pd.read_excel('./data/GSE173083_188.xlsx')
    print("testDataFrame shape:",testDataFrame.shape)
    testDataFrame2=pd.read_csv('./data/new_sequences.csv',encoding='utf-8')
    print("testDataFrame2 shape:",testDataFrame2.shape)
    
    testSeq1 = np.array(testDataFrame['RNA sequence'].values.tolist())
    testSeq2 = np.array(testDataFrame2['sequence'].values.tolist())
    str_ori=np.array(testDataFrame2['structure'].values.tolist())
    loop_ori=np.array(testDataFrame2['bpRNA_string'].values.tolist())
    str=[]
    loop=[]
    for i in range(len(testSeq1)):
        flag=False
        for j in range(len(testSeq2)):
            if testSeq1[i]==testSeq2[j]:
                str.append(str_ori[j])
                loop.append(loop_ori[j])
                flag=True
                break
        if flag==False:
            print(i)
    str=np.array(str)
    loop=np.array(loop)
    testDataFrame['structure']=str
    testDataFrame['loop']=loop
    testDataFrame.to_csv("./data/GSE173083_188_withstrloop.csv")

main()
    

    

