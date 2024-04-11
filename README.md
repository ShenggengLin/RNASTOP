# RNADegpre

This repository contains codes, data and trained models for RNADegpre. RNADegpre is a deep learning model based on large language models and the dual-branch feature decoupling-and-aggregating network to predict mRNA degradation at both the nucleotide resolution and full-length levels.You can find more details about RNADegpre in our paper, "Prediction of mRNA degradation and codon optimization to enhance mRNA stability via deep learning and heuristic search" (Lin et al., 2024).
![image](https://github.com/ShenggengLin/RNADegpre/blob/main/pictures/Model_Architecture_and_Sequence_Optimization.tif)
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

## Configure arnie
In the RNADegpre model, it is necessary to predict the secondary structure of mRNA based on its sequence. In the process of mRNA sequence optimization, it is also necessary to calculate the minimum free energy of mRNA. Both processes require the use of the arnie library. arnie library is a python API to compute RNA energetics and do structure prediction across multiple secondary structure packages. please download the arnie library from https://github.com/DasLab/arnie and configure it according to the instructions. Then place the arnie library under the scripts directory, just like the current code directory.
## Model train and test on the OpenVaccine Kaggle competition dataset
```
cd ./RNADegpre/scripts
conda activate RNADegpre_env
python RNADegpre_train.py
```
The input to RNADegpre_train.py is the OpenVaccine Kaggle competition dataset. RNADegpre_train.py will output the train process and MCRMSE of the training set and test set. The best model parameters will be saved. In the code file, you can modify the model's hyperparameters and training data. And there are detailed comments for each function in the code file.

## Model test on the dataset of full-length mRNAs
```
cd ./RNADegpre/scripts
conda activate RNADegpre_env
python mRNA_deg_spe_cor_dataproce.py
python mRNA_deg_spe_cor.py
```
In order to test the performance of the RNADegpre model on the dataset of full-length mRNAs, you need to first use mRNA_deg_spe_cor_dataproce.py to preprocess the dataset of full-length mRNAs, and get the GSE173083_188_withstrloop.csv file, and then use the GSE173083_188_withstrloop.csv file as the input of mRNA_deg_spe_cor.py to test the model performance. The output of mRNA_deg_spe_cor.py is the correlation coefficient between the predicted value predicted by the model and the experimental value.

## Large language models assist RNADegpre in transfer learning
```
cd ./RNADegpre/scripts
conda activate RNADegpre_env
python RNADegpre_train.py
python RNADegper_withoutpretrain_train.py
python LLMs_assist_transfer_learning.py
```
In order to verify whether the large language model can assist RNADegpre in transfer learning, you need to first run RNADegpre_train.py to get the parameters of the RNADegpre model, then run RNADegper_withoutpretrain_train.py to get the parameters of the RNADegpre model without the large language model, and finally run LLMs_assist_transfer_learning.py to obtain the experimental results in the paper. The output of LLMs_assist_transfer_learning.py is the sequence similarity values and the MCRMSE values of the model in three different cases, as well as the t-SNE plot of the sequences distribution.

## RNADegpre is capable of capturing sequence and structure patterns affecting mRNA degradation

Obtain the results of Figure 4a-g
```
cd ./RNADegpre/scripts
conda activate RNADegpre_env
python best_mse_analysis.py
```
The input to best_mse_analysis.py is the test database of the OpenVaccine Kaggle competition dataset. In the code file, you can modify the RNA data to be predicted. If there are multiple RNAs, the code will sort the RNAs from small to large according to the value of MCRMSE and output them. And there are detailed comments for each function in the code file.

Obtain the results of Figure 4h-i
```
cd ./RNADegpre/scripts
conda activate RNADegpre_env
python vision_motif_3u.py
python vision_loop_motif.py
```
The outputs of vision_motif_3u.py and vision_loop_motif.py are normalized degradation coefficients at different positions in the mRNA sequence.

## RNADegpre is capable of capturing important features affecting mRNA degradation
```
cd ./RNADegpre/scripts
conda activate RNADegpre_env
python RNADegpre_train_without***.py
```
Just modify the definition of RNADegpre in RNADegpre_train.py. Verify the importance of different features by removing different modules.
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
The inputs of beam_rnaopt_covid19.py and MCTS_rnaopt_covid19.py are mRNA sequences to be optimized, such as COVID-19 vaccine sequences. In the code file, you can modify the RNA sequence to be optimized. The code will output the optimization process and the optimized sequence. And there are detailed comments for each function in the code file.

## Note
Some of the drawing programs used in this study are also provided in the scripts directory.
