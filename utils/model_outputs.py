from sklearn.metrics import f1_score
from utils.config import *
from utils.keras_models import *
import numpy as np

class Seq2seqModelOutput:

    def __init__(self, data):

        if data == 'Sentiment':
            self.Config = SentimentConfig()
        elif data == 'Power':
            self.Config = PowerConfig()
        else:
            raise ValueError('Invalid value for data option')
        
    def ensembleoutput(self, x_val, y_val, x_test, y_test, model_folder, model_func, idx):

        # create dictionary for getting label
        dict_ = dict((tuple(x.tolist()), i) for (i, x) in enumerate(idx))

        print('transform labels')
        y_val.resize((y_val.shape[0], y_val.shape[1]*y_val.shape[2]))
        y_test.resize((y_test.shape[0], y_test.shape[1] * y_test.shape[2]))
        y_val = [dict_[tuple(x.tolist())] for x in y_val]
        y_test = [dict_[tuple(x.tolist())] for x in y_test]

        # set decoder inputs
        decoder_val = np.zeros((len(y_val), 1, self.Config.NUM_CLASSES))
        decoder_test = np.zeros((len(y_test), 1, self.Config.NUM_CLASSES))

        model = model_func()

        # predict on model for validation data
        print('predict on validation data')
        pred_sum = np.zeros((len(y_val), self.Config.DECODESTEPS, self.Config.NUM_CLASSES))
        for i in range(self.Config.NUM_ENSEMBLES):
            model.load_weights(model_folder + 'seq_ensem' + str(i) + '.h5')
            pred = model.predict([x_val, decoder_val], batch_size=self.Config.BATCH_SIZE)
            pred_sum += pred

        pred_avg = pred_sum/float(self.Config.NUM_ENSEMBLES)

        # format the prediction and label
        pred = np.round(pred_avg)
        pred.resize((pred.shape[0], pred.shape[1]*pred.shape[2]))
        y_pred = [dict_[tuple(x.tolist())] for x in pred]

        val_f1 = [np.mean(f1_score(y_val, y_pred, average=None)[1:])]

        np.save(model_folder + 'val_lab.npy', y_val)
        np.save(model_folder + 'val_pred.npy', pred)

        # predict on model for test data
        print('predict on test data')
        pred_sum = np.zeros((len(y_test), self.Config.DECODESTEPS, self.Config.NUM_CLASSES))
        for i in range(self.Config.NUM_ENSEMBLES):
            model.load_weights(model_folder + 'seq_ensem' + str(i) + '.h5')
            pred = model.predict([x_test, decoder_test], batch_size=self.Config.BATCH_SIZE)
            pred_sum += pred

        pred_avg = pred_sum/float(self.Config.NUM_ENSEMBLES)
        pred = np.round(pred_avg)
        pred.resize((pred.shape[0], pred.shape[1] * pred.shape[2]))
        y_pred = [dict_[tuple(x.tolist())] for x in pred]

        test_f1 = [np.mean(f1_score(y_test, y_pred, average=None)[1:])]

        np.save(model_folder + 'test_lab.npy', y_test)
        np.save(model_folder + 'test_pred.npy', pred)

        # save outputs
        np.save(model_folder+'ensem_test_f1.npy', test_f1)
        np.save(model_folder+'ensem_val_f1.npy', val_f1)

        return


class Seq2oneModelOutput:

    def __init__(self, data):
        
        if data == 'Sentiment':
            self.Config = SentimentConfig()
        elif data == 'Power':
            self.Config = PowerConfig()
        else:
            raise ValueError('Invalid value for data option')

    # go to ensembling output method
    def ensembleoutput(self, x_val, y_val, x_test, y_test, model_folder, model_func):

        print('load data')
        # load validation data and labels
        if len(y_val.shape) == 3:
            y_val = y_val[:, -1, :]
        y_val = y_val.argmax(axis=1)

        # load test data and labels
        if len(y_test.shape) == 3:
            y_test = y_test[:, -1, :]
        y_test = y_test.argmax(axis=1)

        # set decoder inputs
        decoder_val = np.zeros((len(y_val), 1, self.Config.NUM_CLASSES))
        decoder_test = np.zeros((len(y_test), 1, self.Config.NUM_CLASSES))


        model = model_func()

        # predict on model for validation data
        print('predict on validation data')
        pred_sum = np.zeros((len(y_val), self.Config.NUM_CLASSES))
        for i in range(self.Config.NUM_ENSEMBLES):
            model.load_weights(model_folder + 'seq_ensem' + str(i) + '.h5')
            pred = model.predict([x_val, decoder_val], batch_size=self.Config.BATCH_SIZE)
            pred.resize((len(pred), self.Config.NUM_CLASSES))
            pred_sum += pred

        pred_avg = pred_sum/float(self.Config.NUM_ENSEMBLES)
        pred = np.argmax(pred_avg, axis=1)

        if self.Config.NUM_CLASSES == 2:
            val_f1 = [f1_score(y_val, pred)]
        else:
            val_f1 = [np.mean(f1_score(y_val, pred, average=None)[1:])]

        np.save(model_folder + 'val_lab.npy', y_val)
        np.save(model_folder + 'val_pred.npy', pred)

        # predict on model for test data
        print('predict on test data')
        pred_sum = np.zeros((len(y_test), self.Config.NUM_CLASSES))
        for i in range(self.Config.NUM_ENSEMBLES):
            model.load_weights(model_folder + 'seq_ensem' + str(i) + '.h5')
            pred = model.predict([x_test, decoder_test], batch_size=self.Config.BATCH_SIZE)
            pred.resize((len(pred), self.Config.NUM_CLASSES))
            pred_sum += pred

        pred_avg = pred_sum/float(self.Config.NUM_ENSEMBLES)
        pred = np.argmax(pred_avg, axis=1)

        if self.Config.NUM_CLASSES == 2:
            test_f1 = [f1_score(y_test, pred)]
        else:
            test_f1 = [np.mean(f1_score(y_test, pred, average=None)[1:])]

        np.save(model_folder + 'test_lab.npy', y_test)
        np.save(model_folder + 'test_pred.npy', pred)

        # save outputs
        np.save(model_folder+'ensem_test_f1.npy', test_f1)
        np.save(model_folder+'ensem_val_f1.npy', val_f1)

        return