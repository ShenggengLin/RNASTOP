# RNADegpre

This repository contains codes, data and trained models for RNADegpre. RNADegpre is a deep learning model based on large language models to predict mRNA degradation at the nucleotide resolution.You can find more details about RNADegpre in our paper, "Accurate prediction of RNA degradation at single-nucleotide resolution with large language models" (Lin et al., 2024).
![image](https://github.com/ShenggengLin/RNADegpre/blob/main/pictures/Model_architecture.png)
## Create Environment with Conda

First, download the repository and create the environment.

```
git clone https://github.com/ShenggengLin/RNADegpre.git
cd ./RNADegpre
conda create -n RNADegpre_env
conda activate RNADegpre_env
conda install -r requirements.txt
```

## Configure RNA-FM, DNABERT Pre-training Model

Download the RNA-FM model from https://github.com/ml4bio/RNA-FM and configure it according to the instructions. Then place the RNA-FM-main directory under the scripts directory, just like the current code directory.

Download the DNABERT model weights from https://drive.google.com/drive/folders/1nzlKD29vTcI_3bNPcfjEHT4jYCaW_Ae6?usp=sharing and put it in the scripts/DNABERT6 directory.

## Model Train
```
cd ./RNADegpre/scripts
conda activate RNADegpre_env
python RNADegpre_train.py
```
In the code file, you can modify the model's hyperparameters and training data.

## Model Test
```
cd ./RNADegpre/scripts
conda activate RNADegpre_env
python best_mse_analysis.py
```
In the code file, you can modify the RNA data to be predicted. If there are multiple RNAs, the code will sort the RNAs from small to large according to the value of MCRMSE and output them.

## RNA-seq Optimization
```
#beam search
cd ./RNADegpre/scripts
conda activate RNADegpre_env
python beam_rnaopt_covid19.py

#MCTS
cd ./RNADegpre/scripts
conda activate RNADegpre_env
python MCTS_rnaopt_covid19.py
```
In the code file, you can modify the RNA sequence to be optimized. The code will output the optimization process and the optimized sequence.
