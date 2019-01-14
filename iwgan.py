import sys
from utils.tf_models import *
import numpy as np

'''
Use this script to train GAN with Autoencoder on data (either majority or minority) where labels are not sequences. 

Inputs:
x_train: training data file for GAN model
data_set: name to give training output folder
data: data type to invoke the correct Config file, either 'Power' for power dataset or 'Sentiment' for sentiment dataset
'''

x_train = np.load(sys.argv[1])
data_set = sys.argv[2]
data = sys.argv[3]

save_folder = '../'+data_set+'_iwgan_out/'

iwgan = IWGAN(data)
model_train = iwgan.iwganAeTrain(x_train, save_folder)
