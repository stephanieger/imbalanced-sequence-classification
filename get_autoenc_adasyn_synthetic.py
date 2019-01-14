import sys
from utils.keras_models import Autoencoder
import glob

'''
This script runs the ADASYN algorithm on the trained autoencoder from run_autoenc.py to generate synthetic minority 
data.

Inputs:
ensem_folder: folder where training data is located
model_folder: folder where the trained model is located
data_set: name to give folder with ADASYN synthetic data
data: data type to invoke the correct Config file, either 'Power' for power dataset or 'Sentiment' for sentiment dataset
'''

# get data
ensem_folder = sys.argv[1]
model_folder = sys.argv[2]
model_h5 = max(glob.glob(model_folder+'model??.h5'))
data_set = sys.argv[3]
data = sys.argv[4]
save_folder = '../'+data_set+'_autoenc_syn_adasyn_ensem/'

# generate synthetic data with SMOTE
autoencoder = Autoencoder(data)
autoencoder.runAdasyn(ensem_folder, model_h5, save_folder)
