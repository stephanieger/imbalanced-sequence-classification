import sys
from utils.config import *
from utils.model_outputs import Seq2oneModelOutput
from utils.keras_models import Seq2oneModel
import numpy as np

'''
This script is used to get the validation and test F1 score on a trained seq2one model. Validation and test F1-scores 
are saved as numpy arrays in the model_folder.

Inputs:
model_folder: folder where the trained model is located
data_folder: folder where training data is located
data: data type to invoke the correct Config file, either 'Power' for power dataset or 'Sentiment' for sentiment dataset
'''

# load files
model_folder = sys.argv[1]
data_folder = sys.argv[2]
x_val = np.load(data_folder+'x_val.npy')
y_val = np.load(data_folder+'y_val.npy')
x_test = np.load(data_folder+'x_test.npy')
y_test = np.load(data_folder+'y_test.npy')
data = sys.argv[3]

if data == 'Sentiment':
    Config = SentimentConfig()
elif data == 'Power':
    Config = PowerConfig()
else:
    raise ValueError('Invalid value for data option')


# build model
seq2one = Seq2oneModel(data)
model = seq2one.seq2oneModel()

# model trainer
getoutput = Seq2oneModelOutput(data)
getoutput.ensembleoutput(x_val,y_val, x_test, y_test, model_folder, seq2one.seq2oneModel)