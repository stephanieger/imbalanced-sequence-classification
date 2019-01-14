import sys
import numpy as np
from utils.train_models import AutoencoderTraining
from utils.keras_models import Autoencoder

'''
This script trains an autoencoder for the autoencoder with adasyn oversampling method.

Inputs:
x_train: training data, typically a set of minority data
data_set: name to give training output folder
data: data type to invoke the correct Config file, either 'Power' for power dataset or 'Sentiment' for sentiment dataset
'''

# get data
x_train = np.load(sys.argv[1])
data_set = sys.argv[2]
data = sys.argv[3]
save_folder = '../'+data_set+'_autoenc-out/'

model= Autoencoder(data)
autoencoder, hidden = model.Autoencoder()
print(autoencoder.summary())

train = AutoencoderTraining(data)
train.trainAutoenc(autoencoder, x_train, save_folder)



