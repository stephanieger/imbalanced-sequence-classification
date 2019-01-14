import os
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from utils.config import *
from sklearn.model_selection import train_test_split

# train seq2seq model
class Seq2seqTraining:

    def __init__(self, data):

        if data == 'Sentiment':
            self.Config = SentimentConfig()
        elif data == 'Power':
            self.Config = PowerConfig()
        else:
            raise ValueError('Invalid value for data option')

    def runSeq2seqEnsemble(self, model_func, train_folder, x_val, y_val, save_folder):

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        y_val = y_val.argmax(axis=2)

        decoder_val = np.zeros((len(y_val), 1, self.Config.NUM_CLASSES))

        # get weight labels for data
        weights = class_weight.compute_class_weight('balanced', np.unique(np.max(y_val, axis=1)),
                                                    np.max(y_val, axis=1))

        models = []
        model = model_func()
        for i in range(self.Config.NUM_ENSEMBLES):
            models += [model]

        # train model
        for j in range(self.Config.NUM_ENSEMBLES):
            print('ensemble', j)
            train_dat = np.load(train_folder + 'dat' + str(j) + '.npy')
            train_lab = np.load(train_folder + 'lab' + str(j) + '.npy')

            decoder_train = np.zeros((len(train_lab), 1, self.Config.NUM_CLASSES))

            # store accuracy and loss
            accuracy = -1
            val_f1 = []
            val_loss = []
            train_f1 = []

            for i in range(self.Config.EPOCHS):
                print('epoch', i)
                # train model
                models[j].fit([train_dat, decoder_train], train_lab, batch_size=self.Config.BATCH_SIZE,
                              epochs=1, verbose=1, class_weight=weights)

                # predict on model to get val and train f1
                pred = models[j].predict([train_dat, decoder_train], batch_size=self.Config.BATCH_SIZE)
                train_pred = pred.argmax(axis=2)
                np.save(save_folder + 'train_pred.npy', pred)
                np.save(save_folder + 'train_lab.npy', train_lab)

                pred = models[j].predict([x_val, decoder_val], batch_size=self.Config.BATCH_SIZE)
                np.save('val_pred.npy', pred)
                val_pred = pred.argmax(axis=2)

                # get f1-scores and loss
                if self.Config.NUM_CLASSES == 2:
                    train_f1 += [np.mean(f1_score(train_lab.argmax(axis=2), train_pred, average=None)[1:])]
                    val_f1 += [np.mean(f1_score(y_val, val_pred, average=None)[1:])]
                else:
                    train_f1 += [np.mean(f1_score(train_lab.argmax(axis=2), train_pred,
                                                  labels=[0, 1, 2], average=None)[1:])]
                    val_f1 += [np.mean(f1_score(y_val, val_pred, labels=[0, 1, 2], average=None)[1:])]
                val_loss += [log_loss(y_val, val_pred)]
                # np.save(save_folder + 'val_pred.npy', val_pred)
                np.save(save_folder + 'val_lab.npy', y_val)

                # save model weights, loss and f1-score if validation accuracy improves
                if val_f1[-1] > accuracy:
                    accuracy = val_f1[-1]

                    print('Saved model to disk')
                    model_json = models[j].to_json()
                    with open(save_folder + "seq_ensem" + str(j) + ".json", "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    models[j].save_weights(save_folder + "seq_ensem" + str(j) + ".h5")

                    # save weights, gives us a sense of where the model left off
                    np.save(save_folder + 'ensem_' + str(j) + '_val_loss.npy', val_loss)
                    np.save(save_folder + 'ensem_' + str(j) + '_val_fscore.npy', val_f1)

                np.save(save_folder + 'ensem_' + str(j) + '_val_loss.npy', val_loss)
                np.save(save_folder + 'ensem_' + str(j) + '_val_fscore.npy', val_f1)
                np.save(save_folder + 'ensem_' + str(j) + '_train_fscore.npy', train_f1)

        return


# train sequence to one models
class Seq2oneTraining:
    
    def __init__(self, data):

        if data == 'Sentiment':
            self.Config = SentimentConfig()
        elif data == 'Power':
            self.Config = PowerConfig()
        else:
            raise ValueError('Invalid value for data option')

    # train model with ensembles

    def runSeq2oneEnsemble(self, model_func, train_folder, x_val, y_val, save_folder):

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # load data
        if len(y_val.shape) == 3:
            y_val = y_val[:, -1, :]
        y_val = y_val.argmax(axis=1)

        decoder_val = np.zeros((len(y_val), 1, self.Config.NUM_CLASSES))

        # get weight labels for data
        weights = class_weight.compute_class_weight('balanced', np.unique(y_val), y_val)

        models = []
        model = model_func()
        for i in range(self.Config.NUM_ENSEMBLES):
            models += [model]

        # train model
        for j in range(self.Config.NUM_ENSEMBLES):
            print('ensemble', j)
            train_dat = np.load(train_folder+'dat'+str(j)+'.npy')
            train_lab = np.load(train_folder+'lab'+str(j)+'.npy')
            if len(train_lab.shape) == 3:
                train_lab = train_lab[:, -1, :]
            train_lab = np.expand_dims(train_lab, axis=1)

            decoder_train = np.zeros((len(train_lab), 1, self.Config.NUM_CLASSES))

            # store accuracy and loss
            accuracy = -1
            val_f1 = []
            val_loss = []
            train_f1 = []

            for i in range(self.Config.EPOCHS):
                print('epoch', i)
                # train model
                models[j].fit([train_dat, decoder_train], train_lab, batch_size=self.Config.BATCH_SIZE,
                              epochs=1, verbose=1, class_weight=weights)

                # predict on model to get val and train f1
                pred = models[j].predict([train_dat, decoder_train], batch_size=self.Config.BATCH_SIZE)
                train_pred = pred.argmax(axis=2)
                np.save(save_folder + 'train_pred.npy', pred)
                np.save(save_folder + 'train_lab.npy', train_lab)

                pred = models[j].predict([x_val, decoder_val], batch_size=self.Config.BATCH_SIZE)
                np.save('val_pred.npy', pred)
                val_pred = pred.argmax(axis=2)

                # get f1-scores and loss
                if self.Config.NUM_CLASSES == 2:
                    train_f1 += [f1_score(train_lab.argmax(axis=2), train_pred)]
                    val_f1 += [f1_score(y_val, val_pred)]
                else:
                    train_f1 += [np.mean(f1_score(train_lab.argmax(axis=2), train_pred,
                                                  labels=[0, 1, 2], average=None)[1:])]
                    val_f1 += [np.mean(f1_score(y_val, val_pred, labels=[0, 1, 2], average=None)[1:])]
                val_loss += [log_loss(y_val, np.squeeze(pred, 1))]

                # save model weights, loss and f1-score if validation accuracy improves
                if val_f1[-1] > accuracy:
                    accuracy = val_f1[-1]

                    print('Saved model to disk')
                    model_json = models[j].to_json()
                    with open(save_folder+"seq_ensem" + str(j) + ".json", "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    models[j].save_weights(save_folder+"seq_ensem" + str(j) + ".h5")

                    # save weights, gives us a sense of where the model left off
                    np.save(save_folder +'ensem_' + str(j) + '_val_loss.npy', val_loss)
                    np.save(save_folder +'ensem_' + str(j) + '_val_fscore.npy', val_f1)

                np.save(save_folder + 'ensem_' + str(j) + '_val_loss.npy', val_loss)
                np.save(save_folder + 'ensem_' + str(j) + '_val_fscore.npy', val_f1)
                np.save(save_folder + 'ensem_' + str(j) + '_train_fscore.npy', train_f1)

        return


# train autoencoder models
class AutoencoderTraining:

    def __init__(self, data):

        if data == 'Sentiment':
            self.Config = SentimentConfig()
        elif data == 'Power':
            self.Config = PowerConfig()
        else:
            raise ValueError('Invalid value for data option')

    def trainAutoenc(self, model, x_train, save_folder):

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # use this to keep track of loss
        loss = 100000
        val_loss = []
        train_loss = []

        # transform data
        min_max_scaler = MinMaxScaler()
        train_scale = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        y_train = min_max_scaler.fit_transform(train_scale)
        y_train.resize(x_train.shape)

        # get validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        print(x_train.shape)
        print(x_val.shape)

        # train model
        for i in range(self.Config.EPOCHS):

            hist = model.fit(x_train, y_train, batch_size=self.Config.BATCH_SIZE,
                             validation_data=(x_val, y_val),
                             epochs=1, verbose=1)
            val_loss += [hist.history['val_loss'][0]]
            train_loss += [hist.history['loss'][0]]

            if loss > val_loss[-1]:
                loss = val_loss[-1]

                print("Saved model to disk")

                # serialize model to json
                model_json = model.to_json()
                with open(save_folder + "model" + str(i) + ".json", "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights(save_folder + "model" + str(i) + ".h5")

                # save weights, gives us a sense of where the model left off
                np.save(save_folder + 'val_loss.npy', val_loss)
                np.save(save_folder + 'train_loss.npy', train_loss)

        np.save(save_folder + 'val_loss.npy', val_loss)
        np.save(save_folder + 'train_loss.npy', train_loss)

        return
