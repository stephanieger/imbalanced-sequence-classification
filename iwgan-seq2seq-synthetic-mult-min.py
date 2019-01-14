import sys
from utils.tf_models_seq2seq import *
import numpy as np

'''
Use this script to generate synthetic data on minority given a GAN model trained on minority for sequence label 
vectors.

Inputs:
data_dir: directory where training data is located
data: data type to invoke the correct Config file, either 'Power' for power dataset or 'Sentiment' for sentiment dataset
data_set: name to give folder where training data with GAN synthetic data will be located
checkpoint: saved model weights in trained GAN model folder to use for data generation
epoch: epoch of GAN training where weights for model generation are taken from
fake_real_flag: either 'FAKE' to add noise to generator during data generation or 'REAL' to generate data without added
noise
'''

data_dir = sys.argv[1]
data = sys.argv[2]
data_set = sys.argv[3]
checkpoint = sys.argv[4]
epoch = sys.argv[5]
fake_real_flag = sys.argv[6]

y_min = np.load(data_dir+'y_min.npy')
gen_ensem_save_dir = '../'+fake_real_flag+'_'+data_set+'_iwgan_syn/'
ensem_dir = data_dir+'ensem_'
save_dir = '../'+fake_real_flag+'_'+data_set+'_'+epoch+'_iwgan_syn_ensem/'
syn_dir = gen_ensem_save_dir

iwgan = seq2seqIWGAN(data)
iwgan.iwganGenEnsemFolder(data_dir, gen_ensem_save_dir, checkpoint, fake_real_flag)
iwgan.integrateSynthetic(ensem_dir, syn_dir, save_dir)