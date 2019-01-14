import sys
from utils.tf_models import *
import numpy as np
import os

'''
Use this script to do novelty detection with GAN autoencoder once a GAN is trained on majority data. 

Inputs:
data_dir: directory where training/test/validation data is saved
data_set: name to give training output folder
data: data type to invoke the correct Config file, either 'Power' for power dataset or 'Sentiment' for sentiment dataset
checkpoint: saved model weights in trained GAN model folder to use for data generation
epoch: epoch of GAN training where weights for model generation are taken from
'''

data_dir = sys.argv[1]
data = sys.argv[2]
data_set = sys.argv[3]
checkpoint = sys.argv[4]
epoch = sys.argv[5]

# load relevant data from the data directory
x_test = np.load(data_dir+'x_test.npy')
y_test = np.load(data_dir+'y_test.npy')
x_val = np.load(data_dir+'x_val.npy')
y_val = np.load(data_dir+'y_val.npy')

# make save directory
save_dir = '../'+data_set+'_iwgan_novelty/' + epoch + '/'

# get iwgan mode
iwgan = IWGAN(data)
iwgan.iwganNovelty(save_dir, x_val, y_val, x_test, y_test, checkpoint)
