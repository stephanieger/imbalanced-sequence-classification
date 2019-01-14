import sys
from utils.tf_models_seq2seq import *
import numpy as np

'''
Use this script to train GAN with Autoencoder on data (either majority or minority) for sequence labels.

Inputs:
x_train: training data file for GAN model 
y_train: training label file for GAN model 
data_set: name to give training output folder
data: data type to invoke the correct Config file, either 'Power' for power dataset or 'Sentiment' for sentiment dataset
'''

x_train = np.load(sys.argv[1])
y_train = np.load(sys.argv[2])
data_set = sys.argv[3]
data = sys.argv[4]

save_folder = '../'+data_set+'_iwgan_out/'

iwgan = seq2seqIWGAN(data)
model_train = iwgan.iwganAeTrain(x_train, y_train, save_folder)
