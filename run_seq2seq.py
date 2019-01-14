import sys
import numpy as np
seed_ = int(sys.argv[4])
from numpy.random import seed
seed(seed_)
from tensorflow import set_random_seed
set_random_seed(seed_)
from utils.config import *
from utils.train_models import Seq2seqTraining
from utils.keras_models import Seq2seqModel

'''
This script trains a seq2seq model on ensembled training data. 

Inputs:
train_folder: folder where training data is located
data_set: name to give training output folder
data: data type to invoke the correct Config file, either 'Power' for power dataset or 'Sentiment' for sentiment dataset
seed: set seed for tensorflow and numpy 
'''

# load files
train_folder = sys.argv[1]
x_val = np.load(train_folder+'x_val.npy')
y_val = np.load(train_folder+'y_val.npy')
train_data = train_folder + 'ensem_'
data_set = sys.argv[2]
data = sys.argv[3]

if data == 'Sentiment':
    Config = SentimentConfig()
elif data == 'Power':
    Config = PowerConfig()
else:
    raise ValueError('Invalid value for data option')

save_folder = '../'+data_set+'_seq2seq-out-'+str(Config.NUM_LAYERS)+'/'

# build model
seq2seq = Seq2seqModel(data)
model = seq2seq.seq2seqModel()

# model trainer
train = Seq2seqTraining(data)
train.runSeq2seqEnsemble(seq2seq.seq2seqModel, train_data, x_val, y_val, save_folder)
