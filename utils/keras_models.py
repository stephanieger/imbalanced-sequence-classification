from keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout, RepeatVector, Lambda, GRU
from keras.models import *
from numpy.random import randint, rand
from sklearn.neighbors import NearestNeighbors
from utils.config import *
from utils.AttentionWithContext import AttentionWithContext
from imblearn.over_sampling import SMOTE, ADASYN
from utils.AttentionLSTM import AttentionDecoder


class Seq2seqModel:

    def __init__(self, data):

        if data == 'Sentiment':
            self.Config = SentimentConfig()
        elif data == 'Power':
            self.Config = PowerConfig()
        else:
            raise ValueError('Invalid value for data option')

    def seq2seqModel(self):

        inputs = Input(shape=(self.Config.TIMESTEPS, self.Config.DATA_DIM,))

        # define the first layer of the encoder separately
        enc, state_h, state_c = LSTM(self.Config.HIDDEN_NEURONS, return_sequences=True, stateful=False, dropout=0.2,
                                     recurrent_dropout=0.2, activation='relu', return_state=True,
                                     kernel_initializer='orthogonal')(inputs)


        # iteratively add total number of layers to encoder
        for i in range(self.Config.NUM_LAYERS - 1):
            enc, state_h, state_c = LSTM(self.Config.HIDDEN_NEURONS, return_sequences=True, stateful=False,
                                         dropout=0.2, recurrent_dropout=0.2, activation='relu',
                                         return_state=True, kernel_initializer='orthogonal')(enc)

        # for attention decoder
        decoder_input = Input(shape=(1, self.Config.NUM_CLASSES,))

        decoder_lstm = AttentionDecoder(self.Config.HIDDEN_NEURONS, return_sequences=True,
                                        return_state=True)  # , dropout=0.2, recurrent_dropout=0.2)
        decoder_dropout = Dropout(0.2)
        decoder_dense = Dense(self.Config.NUM_CLASSES, activation='softmax')
        all_outputs = []

        # initialize decoder
        states = [state_h, state_c]
        inputs_ = decoder_input

        for _ in range(self.Config.DECODESTEPS):
            atten, state_h, state_c = decoder_lstm(inputs_, initial_state=states, constants=enc)
            outputs = Lambda(lambda x: K.expand_dims(x, axis=1))(state_h)
            outputs = decoder_dropout(outputs)
            outputs = decoder_dense(outputs)

            # append each output to all_outputs
            all_outputs.append(outputs)

            # reset states
            states = [state_h, state_c]
            inputs_ = outputs

        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        # build model
        model = Model(inputs=[inputs, decoder_input], outputs=[decoder_outputs])

        # define optimizer
        adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0, clipvalue=100)

        # compile model
        model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

        return model


class Seq2oneModel:

    def __init__(self, data):

        if data == 'Sentiment':
            self.Config = SentimentConfig()
        elif data == 'Power':
            self.Config = PowerConfig()
        else:
            raise ValueError('Invalid value for data option')

    # use this to build a model that uses the seq2one model setup
    def seq2oneModel(self):
        # need to define the inputs
        inputs = Input(shape=(self.Config.TIMESTEPS, self.Config.DATA_DIM,))

        # define the first layer of the encoder separately
        enc, state_h, state_c = LSTM(self.Config.HIDDEN_NEURONS, return_sequences=True, stateful=False, dropout=0.2,
                                     recurrent_dropout=0.2, activation='relu', return_state=True,
                                     kernel_initializer='orthogonal')(inputs)

        # iteratively add total number of layers to encoder
        for i in range(self.Config.NUM_LAYERS - 1):
            enc, state_h, state_c = LSTM(self.Config.HIDDEN_NEURONS, return_sequences=True, stateful=False,
                                         dropout=0.2, recurrent_dropout=0.2, activation='relu',
                                         return_state=True, kernel_initializer='orthogonal')(enc)

        # deal with attention now
        atten = AttentionWithContext()(enc)

        encoder_states = [atten, state_c]

        decoder_input = Input(shape=(1, self.Config.NUM_CLASSES))
        decoder_lstm = LSTM(self.Config.HIDDEN_NEURONS, activation='relu', dropout=0.2, recurrent_dropout=0.2,
                            return_sequences=True, kernel_initializer='orthogonal')

        # decode with LSTM and dense layers
        outputs = decoder_lstm(decoder_input, initial_state=encoder_states)
        for i in range(self.Config.NUM_LAYERS-1):
            outputs = Dense(self.Config.DENSE_HIDDEN_NEURONS)(outputs)
            outputs = Dropout(0.2)(outputs)

        outputs = Dense(self.Config.NUM_CLASSES, activation='softmax')(outputs)

        # build model
        model = Model(inputs=[inputs, decoder_input], outputs=[outputs])

        # define optimizer
        adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0, clipvalue=100)

        # compile model
        model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

        return model


class Autoencoder:

    def __init__(self, data):

        if data == 'Sentiment':
            self.Config = SentimentConfig()
        elif data == 'Power':
            self.Config = PowerConfig()
        else:
            raise ValueError('Invalid value for data option')

    def Autoencoder(self):

        # need to define the inputs
        inputs = Input(shape=(self.Config.TIMESTEPS, self.Config.DATA_DIM,))

        # define the first layer of the encoder separately
        enc, state_h, state_c = LSTM(self.Config.HIDDEN_NEURONS, return_sequences=True, stateful=False,
                                     activation='relu', dropout=0.2, recurrent_dropout=0.2,
                                     return_state=True, kernel_initializer='orthogonal')(inputs)

        # iteratively add total number of layers to encoder
        for i in range(self.Config.NUM_LAYERS - 1):
            enc, state_h, state_c = LSTM(self.Config.HIDDEN_NEURONS, return_sequences=True, stateful=False,
                                activation='relu', dropout=0.2, recurrent_dropout=0.2,
                                return_state=True, kernel_initializer='orthogonal')(enc)

        # deal with attention now
        atten = AttentionWithContext()(enc)
        atten_repeat = RepeatVector(self.Config.TIMESTEPS)(atten)

        # deal with decoder now
        decoder_lstm = Bidirectional(LSTM(self.Config.HIDDEN_NEURONS, activation='relu', return_sequences=True, dropout=0.2,
                                          recurrent_dropout=0.2, kernel_initializer='orthogonal'))

        # make decoder
        outputs = decoder_lstm(atten_repeat)
        outputs = Dense(self.Config.DATA_DIM, activation='tanh')(outputs)

        # build autoencoder
        model = Model(inputs=[inputs], outputs=outputs)
        hidden_states = Model(inputs=model.inputs, outputs=atten)

        adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0, clipvalue=100)

        model.compile(loss='mse',
                      optimizer=adadelta,
                      metrics=['accuracy'])

        return model, hidden_states


    # load models and build encoder, decoder and autoencoder
    def loadAutoencoder(self, model_h5):

        # need to define the inputs
        inputs = Input(shape=(self.Config.TIMESTEPS, self.Config.DATA_DIM,))

        # define the first layer of the encoder separately
        enc, state_h, state_c = LSTM(self.Config.HIDDEN_NEURONS, return_sequences=True, stateful=False,
                                     activation='relu', dropout=0.2, recurrent_dropout=0.2,
                                     return_state=True, kernel_initializer='orthogonal')(inputs)

        # iteratively add total number of layers to encoder
        for i in range(self.Config.NUM_LAYERS - 1):
            enc, state_h, state_c = LSTM(self.Config.HIDDEN_NEURONS, return_sequences=True, stateful=False,
                                         activation='relu', dropout=0.2, recurrent_dropout=0.2,
                                         return_state=True, kernel_initializer='orthogonal')(enc)

        # deal with attention now
        atten = AttentionWithContext()(enc)
        atten_repeat = RepeatVector(self.Config.TIMESTEPS)(atten)

        # deal with decoder now
        decoder_lstm = Bidirectional(LSTM(self.Config.HIDDEN_NEURONS, activation='relu', return_sequences=True, dropout=0.2,
                                          recurrent_dropout=0.2, kernel_initializer='orthogonal'))

        # make decoder
        outputs = decoder_lstm(atten_repeat)
        outputs = Dense(self.Config.DATA_DIM, activation='tanh')(outputs)

        # build autoencoder
        model = Model(inputs=[inputs], outputs=outputs)

        adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

        model.compile(loss='mse',
                      optimizer=adadelta,
                      metrics=['accuracy'])

        model.load_weights(model_h5)

        encoder = Model(inputs=inputs, outputs=atten)

        encoder.compile(loss='mse',
                        optimizer=adadelta,
                        metrics=['accuracy'])

        decoder_layers = model.layers[-3:]
        inputs = Input(shape=(self.Config.HIDDEN_NEURONS, ))
        decode = decoder_layers[0](inputs)
        for i in range(1, len(decoder_layers)):
            decode = decoder_layers[i](decode)

        decoder = Model(inputs=[inputs], outputs=decode)

        return model, encoder, decoder

    def integrateSynDat(self, syn_dir, save_dir, data_dir):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(self.Config.NUM_ENSEMBLES):
            print(i)
            e_lab = np.load(data_dir + 'dat' + str(i) + '.npy')
            e_dat = np.load(data_dir + 'lab' + str(i) + '.npy')
            s_lab = np.load(syn_dir + 'synthetic_lab' + str(i) + '.npy')
            s_dat = np.load(syn_dir + 'synthetic_dat' + str(i) + '.npy')

            c_dat = np.concatenate((e_dat, s_dat), axis=0)
            c_lab = np.concatenate((e_lab, s_lab), axis=0)

            shuffle = np.random.choice(len(c_lab), len(c_lab), replace=False)

            np.save(save_dir + 'lab_' + str(i) + '.npy', c_lab[shuffle])
            np.save(save_dir + 'dat' + str(i) + '.npy', c_dat[shuffle])

        return

    def runAdasyn(self, ensem_folder, model_h5, save_dir):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # build and load models
        autoencoder, encoder, decoder = self.loadAutoencoder(model_h5)

        for ensem in range(self.Config.NUM_ENSEMBLES):

            dat = np.load(ensem_folder+'ensem_dat'+str(ensem)+'.npy')
            lab = np.load(ensem_folder+'ensem_lab'+str(ensem)+'.npy')
            dat_ = encoder.predict(dat)

            # resize data
            if len(lab.shape) == 3:
                lab = lab[:, -1, :]
                lab = np.argmax(lab, axis=1)
            else:
                lab = np.argmax(lab, axis=1)

            # run adasyn
            print(ensem)
            print('run ADASYN')

            ada = ADASYN(ratio='minority', random_state=42)

            # fit smote object
            print('fit smote object for ensem ' + str(ensem))
            x_res, y_res = ada.fit_sample(dat_, lab)

            x_syn = decoder.predict(x_res)

            y_res_ = []
            for i in range(len(y_res)):
                if y_res[i] == 0:
                    y_res_ += [np.array([1, 0])]
                else:
                    y_res_ += [np.array([0, 1])]

            y_res_ = np.array(y_res_)

            # save data
            print('save ensem ' + str(ensem))
            np.save(save_dir + 'ensem_dat' + str(ensem) + '.npy', x_syn)
            np.save(save_dir + 'ensem_lab' + str(ensem) + '.npy', y_res_)


        return

