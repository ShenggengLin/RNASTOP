# RNASTOP

This repository contains codes, data and trained models for RNASTOP. RNASTOP is a a novel framework that accurately predicts mRNA degradation at both single-nucleotide and full-length levels, while also enhancing mRNA stability through codon optimization. RNASTOP encompasses an mRNA degradation prediction model and an mRNA codon optimization module. The mRNA degradation prediction model primarily comprises three modules: the mRNA feature embedding module, the dual-branch feature decoupling-and-aggregating network and the prediction layer. Firstly, the model integrates nucleic acid LLMs and the structure embedding module to obtain the multi-source feature embeddings of mRNA sequences. These embeddings are then fed into the dual-branch feature decoupling-and-aggregating network for feature fusion. Finally, the fused feature embeddings are fed into the prediction layer for the prediction of mRNA degradation. mRNA codon optimization module further integrates the mRNA degradation prediction model with beam search for codon optimization to improve the stability of mRNA. You can find more details about RNASTOP in our paper, "RNASTOP: Prediction and optimization of mRNA stability by integrating deep learning and thermodynamic property-guided heuristic search" (Lin et al., 2024).
![image](https://github.com/ShenggengLin/RNASTOP/blob/main/pictures/figure%201.tif)
## Create Environment with Conda

First, download the repository and create the environment.

```
git clone https://github.com/ShenggengLin/RNASTOP.git
cd ./RNASTOP
conda create -n RNASTOP_env
conda activate RNASTOP_env
conda install -r requirements.txt
```

## Configure RNA-FM, DNABERT Pre-training Model

Download the RNA-FM model from https://github.com/ml4bio/RNA-FM and configure it according to the instructions. Then place the RNA-FM-main directory under the scripts directory, just like the current code directory.

Download the DNABERT model weights from https://drive.google.com/drive/folders/1nzlKD29vTcI_3bNPcfjEHT4jYCaW_Ae6?usp=sharing and put it in the scripts/DNABERT6 directory.

## Configure arnie
In the RNASTOP model, it is necessary to predict the secondary structure of mRNA based on its sequence. In the process of mRNA sequence optimization, it is also necessary to calculate the minimum free energy of mRNA. Both processes require the use of the arnie library. arnie library is a python API to compute RNA energetics and do structure prediction across multiple secondary structure packages. please download the arnie library from https://github.com/DasLab/arnie and configure it according to the instructions. Then place the arnie library under the scripts directory, just like the current code directory.
## Model train and test on the OpenVaccine Kaggle competition dataset
```
cd ./RNASTOP/scripts
conda activate RNASTOP_env
python RNASTOP_train.py
```
The input to RNASTOP_train.py is the OpenVaccine Kaggle competition dataset. RNASTOP_train.py will output the train process and MCRMSE of the training set and test set. The best model parameters will be saved. In the code file, you can modify the model's hyperparameters and training data. And there are detailed comments for each function in the code file.

## Model test on the dataset of full-length mRNAs
```
cd ./RNASTOP/scripts
conda activate RNASTOP_env
python mRNA_deg_spe_cor_dataproce.py
python mRNA_deg_spe_cor.py
```
In order to test the performance of the RNASTOP model on the dataset of full-length mRNAs, you need to first use mRNA_deg_spe_cor_dataproce.py to preprocess the dataset of full-length mRNAs, and get the GSE173083_188_withstrloop.csv file, and then use the GSE173083_188_withstrloop.csv file as the input of mRNA_deg_spe_cor.py to test the model performance. The output of mRNA_deg_spe_cor.py is the correlation coefficient between the predicted value predicted by the model and the experimental value.

## Large language models assist RNASTOP in transfer learning
```
cd ./RNASTOP/scripts
conda activate RNASTOP_env
python RNASTOP_train.py
python RNASTOP_withoutpretrain_train.py
python LLMs_assist_transfer_learning.py
```
In order to verify whether the large language model can assist RNASTOP in transfer learning, you need to first run RNASTOP_train.py to get the parameters of the RNASTOP model, then run RNASTOP_withoutpretrain_train.py to get the parameters of the RNASTOP model without the large language model, and finally run LLMs_assist_transfer_learning.py to obtain the experimental results in the paper. The output of LLMs_assist_transfer_learning.py is the sequence similarity values and the MCRMSE values of the model in three different cases, as well as the t-SNE plot of the sequences distribution.

## RNASTOP is capable of capturing sequence and structure patterns affecting mRNA degradation

Obtain the results of Figure 4a-g
```
cd ./RNASTOP/scripts
conda activate RNASTOP_env
python best_mse_analysis.py
```
The input to best_mse_analysis.py is the test database of the OpenVaccine Kaggle competition dataset. In the code file, you can modify the RNA data to be predicted. If there are multiple RNAs, the code will sort the RNAs from small to large according to the value of MCRMSE and output them. And there are detailed comments for each function in the code file.

Obtain the results of Figure 4h-i
```
cd ./RNASTOP/scripts
conda activate RNASTOP_env
python vision_motif_3u.py
python vision_loop_motif.py
```
The outputs of vision_motif_3u.py and vision_loop_motif.py are normalized degradation coefficients at different positions in the mRNA sequence.

## RNASTOP is capable of capturing important features affecting mRNA degradation
```
cd ./RNASTOP/scripts
conda activate RNASTOP_env
python RNASTOP_train_without***.py
```
Just modify the definition of RNASTOP in RNASTOP_train.py. Verify the importance of different features by removing different modules.
## RNA-seq Optimization
```
#beam search
cd ./RNASTOP/scripts
conda activate RNASTOP_env
python beam_rnaopt_covid19.py

#MCTS
cd ./RNASTOP/scripts
conda activate RNASTOP_env
python MCTS_rnaopt_covid19.py
```
The inputs of beam_rnaopt_covid19.py and MCTS_rnaopt_covid19.py are mRNA sequences to be optimized, such as COVID-19 vaccine sequences. In the code file, you can modify the RNA sequence to be optimized. The code will output the optimization process and the optimized sequence. And there are detailed comments for each function in the code file.

## Note
Some of the drawing programs used in this study are also provided in the scripts directory.

## License
The software is released under MIT License.

Copyright (c) 2022 Kai.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
