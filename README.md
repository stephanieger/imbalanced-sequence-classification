This repository contains implementations of the models discussed in the paper 
["Autoencoders and Generative Adversarial Networks for Anomaly Detection for Sequences"](https://arxiv.org/abs/1901.02514)
by Stephanie Ger and Diego Klabjan. 

## Table of Contents
* Data
* Baseline Models
* GAN Based Models
* ADASYN with Autoencoder Models

## Data 
Models were evaluated on two public datasets and these datasets are available [here](https://northwestern.box.com/s/lt1mkyjhbl0ksq21y1m0o9qpkd6g5ib5). The file norm-sentiment-0.01.tar.gz refers to the sentiment dataset with 1% imbalance and the norm-sentiment-0.05.tar.gz is the sentiment dataset with 5% imbalance. The files with power in the filename contain the power datasets. We provide ensembled power datasets with 5 different seeds. Each .zip file contains the ensembled training data, validation and test data. Minority and majority data is also included to train GAN and autoencoder models for the oversampling methods described in the paper. All data files are stored as numpy arrays. 

## Baseline Models
The baseline model is run using the run_seq2one.py or run_seq2seq.py scripts depending on if the label vector is a 
sequence or not. The F1-score for the validation and test sets can be computed using the run_seq2one_output.py and
run_seq2seq.py scripts respectively.

## GAN Models
For novelty detection with either the GAN discriminator or GAN autoencoder as the novelty detection method, first a GAN
is trained on majority data using the iwgan.py script. Then, the two novelty detection methods can be run with the
iwgan-autoenc-novelty.py and iwgan-discrim-novelty.py scripts respectively. 

For GAN based synthetic data generation, a GAN is trained on minority data with the iwgan.py script or iwgan-seq2seq.py
script depending on if the label vector is a sequence or not. Then, synthetic data can be generated with 
iwgan-synthetic-mult-min.py or iwgan-seq2seq-synthetic-mult-min.py respectively and the seq2one or seq2seq model can be
run.

## ADASYN with Autoencoder Models
For ADASYN with Autoencoder, the run_autoenc.py script can be used to train the autoencoder model on the minority data. 
Then get_autoenc_adasyn_synthetic.py can be used to generate the synthetic data. The training set with the synthetic 
data can be used to train a seq2one model with the run_seq2one.py script. 
