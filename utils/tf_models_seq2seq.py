import os
import keras.backend as K
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error as mse
from utils.config import *
from utils.AttentionWithContext import AttentionWithContext
import keras
from utils.AttentionLSTM import AttentionDecoder

# implementation of improved wassertein gan in tensorflow

class seq2seqIWGAN:
    def __init__(self, data):

        if data == 'Sentiment':
            self.Config = SentimentConfig()
        elif data == 'Power':
            self.Config = PowerConfig()
        else:
            raise ValueError('Invalid value for data option')

        self.batchsize = self.Config.GAN_BATCH_SIZE
        self.timesteps = self.Config.TIMESTEPS
        self.decodesteps = self.Config.DECODESTEPS
        self.data_dim = self.Config.DATA_DIM
        self.data = data
        self.num_classes = self.Config.NUM_CLASSES

    # define leakyrelu activation
    def leakyrelu(self, x, alpha=0.3, name='lrelu'):
        return tf.maximum(x, alpha * x)

    # define generator model
    def generator(self, noisy, real_data, real_labels, noise_level, reuse=False):

        with tf.variable_scope('generator', reuse=reuse) as scope:

            # define LSTM with dropout
            def lstm_():
                return rnn.DropoutWrapper(rnn.LSTMCell(self.Config.G_HIDDEN_NEURONS),
                                          input_keep_prob=self.Config.DROPOUT_IN,
                                          output_keep_prob=self.Config.DROPOUT_OUT)

            # define noisy hidden state for fake data
            h_state = tf.random_normal([self.batchsize, self.Config.G_HIDDEN_NEURONS])
            c_state = tf.zeros([self.batchsize, self.Config.G_HIDDEN_NEURONS])

            lstm_state = rnn.LSTMStateTuple(c_state, h_state)


            # define encoder
            # stack layers of LSTM
            enc_stacked_lstm = rnn.MultiRNNCell(
                [lstm_() for _ in range(self.Config.G_NUM_LAYERS)])

            # set initial state based on if it's real data or not
            if noisy == True:
                enc_init_state = [lstm_state for _ in range(self.Config.G_NUM_LAYERS)]
                enc_init_state = tuple(enc_init_state)

            else:
                enc_init_state = enc_stacked_lstm.zero_state(self.batchsize, tf.float32)

            enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_stacked_lstm, real_data, initial_state=enc_init_state, time_major=False,
                                                scope='generator/encoder')

            enc_outputs = self.leakyrelu(enc_outputs)


            # define decoder
            dec_stacked_lstm = rnn.MultiRNNCell(
                [lstm_() for _ in range(self.Config.G_NUM_LAYERS)])

            enc_states = tf.unstack(enc_states, axis=0)
            dec_init_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(enc_states[idx][0], enc_states[idx][1])
                                    for idx in range(self.Config.G_NUM_LAYERS)])

            dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_stacked_lstm, real_labels, initial_state=dec_init_state,
                                                        time_major=False,
                                                        scope='generator/decoder')


            if self.Config.INSTANCE_NOISE:
                enc_output_noise = tf.random_normal([self.batchsize, self.timesteps, self.Config.G_HIDDEN_NEURONS],
                                                stddev=noise_level)
                dec_output_noise = tf.random_normal([self.batchsize, self.decodesteps, self.Config.G_HIDDEN_NEURONS],
                                                    stddev=noise_level)
                enc_outputs += enc_output_noise
                dec_outputs += dec_output_noise

        return enc_outputs, enc_states, dec_outputs, dec_states

    # define discriminator model
    def discriminator(self, enc_inputs, dec_inputs, reuse=False):

        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            # build RNN here
            lstm_ = rnn.LSTMCell(self.Config.D_HIDDEN_NEURONS)
            lstm_ = rnn.DropoutWrapper(lstm_, input_keep_prob=self.Config.DROPOUT_IN,
                                       output_keep_prob=self.Config.DROPOUT_OUT)

            stacked_lstm = rnn.MultiRNNCell(
                [lstm_ for _ in range(self.Config.D_NUM_LAYERS)])

            # initialize initial state to 0
            init_state = stacked_lstm.zero_state(self.batchsize, tf.float32)

            # bidirectional RNN
            enc_outputs, enc_fw_state, enc_bw_state = rnn.static_bidirectional_rnn(stacked_lstm, stacked_lstm,
                                                         tf.unstack(tf.transpose(enc_inputs, perm=[1, 0, 2])),
                                                         initial_state_fw=init_state, initial_state_bw=init_state,
                                                         scope='discriminator/encoder/')

            dec_outputs, _, _ = rnn.static_bidirectional_rnn(stacked_lstm, stacked_lstm,
                                                             tf.unstack(tf.transpose(dec_inputs,perm=[1, 0, 2])),
                                                             initial_state_fw=enc_fw_state,
                                                             initial_state_bw=enc_bw_state,
                                                             scope='discriminator/decoder/')

            outputs = tf.contrib.layers.fully_connected(dec_outputs, self.Config.AE_DENSE_NEURONS, reuse=reuse,
                                                        scope=scope, activation_fn=None)

            # specify activation function here
            outputs = self.leakyrelu(outputs)
            # dropout
            outputs = tf.nn.dropout(outputs, keep_prob=self.Config.DROPOUT_OUT)
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

            # use only the last hidden state as input to dense layer
            outputs = tf.slice(outputs, [0, self.Config.DECODESTEPS - 1, 0], [self.batchsize, 1,
                                                                            self.Config.AE_DENSE_NEURONS])
            # use last hidden state as input to a dense layer
            outputs = tf.layers.dense(outputs, int(self.Config.AE_DENSE_NEURONS / 2), name='discriminator/pre_output')
            # specify activation function here
            outputs = self.leakyrelu(outputs)
            # dropout
            outputs = tf.nn.dropout(outputs, keep_prob=self.Config.DROPOUT_OUT)
            # final output of discriminator
            outputs = tf.layers.dense(outputs, 1, name='discriminator/output')

        return outputs

    # build autoencoder seq2seq model with predictions
    def autoencoder_seq2seq_pred(self, enc_inputs, enc_states, dec_inputs, dec_states, data_in, label_in, reuse=False):

        with tf.variable_scope('autoencoder', reuse=reuse) as scope:
            # get attention from hidden states of generator
            atten = AttentionWithContext()(enc_inputs)

            enc_states = tf.unstack(enc_states, axis=0)
            dec_states = tf.unstack(dec_states, axis=0)
            # set inital cell state to cell state from generator
            # set inital hidden state to attention of hidden states from generator
            enc_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(enc_states[idx][0], atten) for idx in range(self.Config.A_NUM_LAYERS)])
            dec_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(dec_states[idx][0], dec_states[idx][1])
                 for idx in range(self.Config.A_NUM_LAYERS)])

            def lstm_():
                return rnn.DropoutWrapper(rnn.LSTMCell(self.Config.A_HIDDEN_NEURONS),
                                          input_keep_prob=self.Config.DROPOUT_IN,
                                          output_keep_prob=self.Config.DROPOUT_OUT)

            stacked_lstm = rnn.MultiRNNCell(
                [lstm_() for _ in range(self.Config.A_NUM_LAYERS)])

            enc_in = tf.zeros([self.batchsize, 1, self.Config.DATA_DIM])
            dec_in = tf.zeros([self.batchsize, 1, self.Config.NUM_CLASSES])

            all_enc_outputs = []
            all_dec_outputs = []

            # feed in previous output as current input (teacher forcing)
            for j in range(self.Config.TIMESTEPS):
                outputs, states = tf.nn.dynamic_rnn(stacked_lstm, enc_in, initial_state=enc_tuple_state,
                                                    time_major=False,
                                                    scope=scope)
                outputs = self.leakyrelu(outputs)
                outputs = tf.layers.dense(outputs, self.Config.DATA_DIM, name='autoencoder/enc_output')
                # outputs = tf.sigmoid(outputs, name='autoencoder/sigmoid')

                all_enc_outputs += [outputs]
                # ae_in = outputs
                enc_in = tf.expand_dims(data_in[:, j, :], axis=1)

                scope.reuse_variables()

            enc_outputs = tf.concat(all_enc_outputs, 1)

            decoder_lstm = keras.layers.LSTM(self.Config.A_HIDDEN_NEURONS, return_sequences=True,
                                            return_state=True)  # , dropout=0.2, recurrent_dropout=0.2)
            decoder_dense = keras.layers.Dense(self.Config.NUM_CLASSES, activation='softmax')
            dec_states = [dec_states[0][0], dec_states[0][1]]

            for j in range(self.Config.DECODESTEPS):

                outputs, state_h, state_c = decoder_lstm(dec_in, initial_state=dec_states)
                outputs = decoder_dense(outputs)

                all_dec_outputs += [outputs]
                dec_in = tf.expand_dims(label_in[:, j, :], axis=1)
                dec_states = [state_h, state_c]

                scope.reuse_variables()

            dec_outputs = tf.concat(all_dec_outputs, 1)

        return enc_outputs, dec_outputs

    def iwganAeTrain(self, x_train, y_train, save_dir):

        d_loss = []
        d_loss_diff = []
        g_loss = []
        a_loss = []
        g_loss_diff = []

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_in = tf.placeholder(tf.float32, shape=[None, self.timesteps, self.data_dim])
        label_in = tf.placeholder(tf.float32, shape=[None, self.decodesteps, self.num_classes])
        noise = tf.placeholder(tf.float32, shape=())

        fake_enc_data, fake_enc_states, fake_dec_data, fake_dec_states = self.generator(True, data_in,
                                                                                        label_in, noise, reuse=False)
        real_enc_data, real_enc_states, real_dec_data, real_dec_states = self.generator(False, data_in,
                                                                                        label_in, noise, reuse=True)
        disc_fake = self.discriminator(fake_enc_data, fake_dec_data, reuse=False)
        disc_real = self.discriminator(real_enc_data, real_dec_data, reuse=True)

        autoenc_data, autoenc_label = self.autoencoder_seq2seq_pred(real_enc_data, real_enc_states, real_dec_data,
                                                                    real_dec_states, data_in, label_in, reuse=False)

        # calculate loss
        ae_cost = 0
        # data AE loss
        ae_cost += tf.reduce_mean(tf.norm(tf.reshape(data_in, [self.batchsize, self.timesteps * self.data_dim]) -
                                          tf.reshape(autoenc_data, [self.batchsize, self.timesteps * self.data_dim]),
                                          axis=1))
        # label AE loss
        ae_cost += tf.reduce_mean(tf.norm(tf.reshape(label_in, [self.batchsize, self.decodesteps * self.num_classes]) -
                                          tf.reshape(autoenc_label, [self.batchsize, self.decodesteps *
                                                                    self.num_classes]),
                                          axis=1))

        disc_cost_ = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        gen_cost_ = -tf.reduce_mean(disc_fake)
        gen_cost = gen_cost_ + self.Config.MU * ae_cost

        # gradient loss
        alpha = tf.random_uniform(
            shape=[self.batchsize, 1, 1],
            minval=0.,
            maxval=1.
        )
        dec_interpolates = alpha * real_dec_data + ((1 - alpha) * fake_dec_data)
        enc_interpolates = alpha * real_enc_data + ((1 - alpha) * fake_enc_data)
        disc_interpolates = self.discriminator(enc_interpolates, dec_interpolates, reuse=True)
        gradients = tf.gradients(disc_interpolates, [enc_interpolates, dec_interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]) + self.Config.EP)
        gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
        disc_cost = disc_cost_ + self.Config.LAMBDA * gradient_penalty

        ################################################ SET UP MODEL TRAINING #########################################
        print('set up model training')
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'generator' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        a_vars = [var for var in t_vars if 'autoencoder' in var.name]

        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=self.Config.DISC_LEARNING_RATE,
            beta1=0.5,
            beta2=0.9).minimize(disc_cost, var_list=d_vars)

        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=self.Config.LEARNING_RATE,
            beta1=0.5,
            beta2=0.9).minimize(gen_cost, var_list=g_vars)

        autoenc_train_op = tf.train.AdamOptimizer(
            learning_rate=self.Config.LEARNING_RATE,
            beta1=0.5,
            beta2=0.9).minimize(ae_cost, var_list=a_vars)

        ################################################## TRAINING LOOP ###############################################
        print('training loop')
        saver = tf.train.Saver(max_to_keep=None)

        init_op = tf.global_variables_initializer()

        session = tf.Session()
        K.set_session(session)
        session.run(init_op)

        # set initial noise
        _noise = 1

        with session.as_default():

            for epoch in range(self.Config.GAN_AE_EPOCHS):
                print('epoch:', epoch)
                np.random.shuffle(x_train)
                minibatch_size = self.batchsize * (self.Config.DISC_CRITIC_ITERS + self.Config.GEN_CRITIC_ITERS + 1)

                if (epoch + 1) % 100 == 0:
                    if self.data == 'Sentiment':
                        _noise += -0.2

                        if _noise < 0:
                            _noise = 0.

                    elif self.data == 'Power':
                        _noise += -0.2

                        if _noise < 0:
                            _noise = 0.

                    print(_noise)

                for i in range(int(len(x_train) // (self.batchsize * (self.Config.DISC_CRITIC_ITERS +
                                                                          self.Config.GEN_CRITIC_ITERS + 1)))):
                    data_minibatch = x_train[i * minibatch_size: (i + 1) * minibatch_size]
                    label_minibatch = y_train[i * minibatch_size: (i + 1) * minibatch_size]
                    print('minibatch:', i)
                    for j in range(self.Config.GEN_CRITIC_ITERS):
                        _data = data_minibatch[j * self.batchsize: (j + 1) * self.batchsize]
                        _label = label_minibatch[j * self.batchsize: (j + 1) * self.batchsize]
                        _gen_cost, _gen_cost_diff, _ = session.run([gen_cost, gen_cost_, gen_train_op],
                                                                   feed_dict={data_in: _data, label_in: _label,
                                                                              noise: _noise})
                        g_loss += [_gen_cost]
                        g_loss_diff += [_gen_cost_diff]
                    for j in range(self.Config.DISC_CRITIC_ITERS):
                        _data = data_minibatch[(self.Config.GEN_CRITIC_ITERS + j) * self.batchsize:
                        (self.Config.GEN_CRITIC_ITERS + j + 1) * self.batchsize]
                        _label = label_minibatch[(self.Config.GEN_CRITIC_ITERS + j) * self.batchsize:
                        (self.Config.GEN_CRITIC_ITERS + j + 1) * self.batchsize]
                        _disc_cost, _disc_cost_diff, _ = session.run([disc_cost, disc_cost_, disc_train_op],
                                                                     feed_dict={data_in: _data, label_in: _label,
                                                                                noise: _noise})
                        d_loss += [_disc_cost]
                        d_loss_diff += [_disc_cost_diff]

                    _data = data_minibatch[(self.Config.DISC_CRITIC_ITERS + self.Config.GEN_CRITIC_ITERS) *
                                           self.batchsize: (self.Config.GEN_CRITIC_ITERS +
                                                            self.Config.DISC_CRITIC_ITERS + 1) * self.batchsize]
                    _label = label_minibatch[(self.Config.DISC_CRITIC_ITERS + self.Config.GEN_CRITIC_ITERS) *
                                           self.batchsize: (self.Config.GEN_CRITIC_ITERS +
                                                            self.Config.DISC_CRITIC_ITERS + 1) * self.batchsize]
                    _ae_cost, _ = session.run([ae_cost, autoenc_train_op], feed_dict={data_in: _data, label_in: _label,
                                                                                      noise: _noise})
                    a_loss += [_ae_cost]

                if epoch >= 0 and epoch % self.Config.CHECKPOINT_STEP == 0:
                    saver.save(session, save_dir + 'model', global_step=epoch)
                    np.save(save_dir + 'd_loss.npy', d_loss)
                    np.save(save_dir + 'g_loss.npy', g_loss)
                    np.save(save_dir + 'g_diff_loss.npy', g_loss_diff)
                    np.save(save_dir + 'd_diff_loss.npy', d_loss_diff)
                    np.save(save_dir + 'a_loss.npy', a_loss)

            saver.save(session, save_dir + 'model', global_step=self.Config.GAN_AE_EPOCHS - 1)
            np.save(save_dir + 'd_loss.npy', d_loss)
            np.save(save_dir + 'g_loss.npy', g_loss)
            np.save(save_dir + 'g_diff_loss.npy', g_loss_diff)
            np.save(save_dir + 'd_diff_loss.npy', d_loss_diff)
            np.save(save_dir + 'a_loss.npy', a_loss)

        return


    # build a model to generate ensembles of data
    def iwganGenEnsemFolder(self, data_dir, save_dir, checkpoint, flag):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        session = tf.Session()
        K.set_session(session)

        print('calculate loss function')
        # get model outputs
        data_in = tf.placeholder(tf.float32, shape=[None, self.timesteps, self.data_dim])
        label_in = tf.placeholder(tf.float32, shape=[None, self.decodesteps, self.num_classes])
        noise = 0.0

        fake_enc_data, fake_enc_states, fake_dec_data, fake_dec_states = self.generator(True, data_in,
                                                                                        label_in, noise, reuse=False)
        real_enc_data, real_enc_states, real_dec_data, real_dec_states = self.generator(False, data_in,
                                                                                        label_in, noise, reuse=True)

        if flag == 'FAKE':
            autoenc_data, autoenc_label = self.autoencoder_seq2seq_pred(fake_enc_data, fake_enc_states, fake_enc_data,
                                                                        fake_enc_states, data_in, label_in, reuse=False)
        if flag == 'REAL':
            autoenc_data, autoenc_label = self.autoencoder_seq2seq_pred(real_enc_data, real_enc_states, real_enc_data,
                                                                        real_enc_states, data_in, label_in, reuse=False)

        ###################################### LOAD SAVED VARIABLES AND GENERATE DATA ##################################
        # create label dictionary
        dict_idx = np.load(data_dir + 'idx.npy')
        dict_ = dict((tuple(x.tolist()), i) for (i, x) in enumerate(dict_idx))

        print('generate data')

        init_op = tf.global_variables_initializer()
        session.run(init_op)

        saver = tf.train.Saver()

        with session.as_default():

            saver.restore(session, checkpoint)

            print('weights restored')

            for ensem in range(self.Config.AE_NUM_ENSEMBLES):

                dat = np.load(data_dir + 'ensem_dat' + str(ensem) + '.npy')
                lab = np.load(data_dir + 'ensem_lab' + str(ensem) + '.npy')

                lab_ = np.reshape(lab, (lab.shape[0], lab.shape[1]*lab.shape[2]))
                lab_ = [dict_[tuple(x.tolist())] for x in lab_]
                idx = np.where(lab_ != 0)[0]

                x_train = dat[idx]
                print(x_train.shape)
                y_train = lab[idx]
                print('ensemble:', ensem)
                syn_dat = []
                syn_lab = []

                for gen_epochs in range(self.Config.NUM_SYN_ITER):
                    print('iteration through data:', gen_epochs)

                    np.random.shuffle(x_train)
                    if int(len(x_train)) < self.batchsize:
                        x_train = np.repeat(x_train, self.batchsize, axis=0)
                        y_train = np.repeat(y_train, self.batchsize, axis=0)
                        for i in range(int(len(x_train)) // self.batchsize):
                            data_ = x_train[i * self.batchsize: (i + 1) * self.batchsize]
                            label_ = y_train[i * self.batchsize: (i + 1) * self.batchsize]
                            syn_dat += [autoenc_data.eval(feed_dict={data_in: data_, label_in: label_})]
                            syn_lab += [autoenc_label.eval(feed_dict={data_in: data_, label_in: label_})]

                        syn_dat = np.array(syn_dat)
                        syn_lab = np.array(syn_lab)
                        syn_lab = np.round(syn_lab)

                        syn_lab = syn_lab[0::self.batchsize]
                        syn_dat = syn_dat[0::self.batchsize]
                    else:
                        for i in range(int(len(x_train)) // self.batchsize):
                            data_ = x_train[i * self.batchsize: (i + 1) * self.batchsize]
                            label_ = y_train[i * self.batchsize: (i + 1) * self.batchsize]
                            syn_dat += [autoenc_data.eval(feed_dict={data_in: data_, label_in: label_})]
                            syn_lab += [autoenc_label.eval(feed_dict={data_in: data_, label_in: label_})]

                        syn_dat = np.array(syn_dat)
                        syn_lab = np.array(syn_lab)
                        syn_lab = np.round(syn_lab)


                print(syn_lab)
                syn_dat.resize(syn_dat.shape[0] * syn_dat.shape[1], syn_dat.shape[2], syn_dat.shape[3])
                syn_lab.resize(syn_lab.shape[0] * syn_lab.shape[1], syn_lab.shape[2], syn_lab.shape[3])
                np.save(save_dir + 'synthetic_data_' + str(ensem) + '.npy', syn_dat)
                np.save(save_dir + 'synthetic_label_' + str(ensem) + '.npy', syn_lab)
        return

    def iwganGenEnsemListFolder(self, data_dir, ensem_list, save_dir, checkpoint, flag):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        session = tf.Session()
        K.set_session(session)

        print('calculate loss function')
        # get model outputs
        data_in = tf.placeholder(tf.float32, shape=[None, self.timesteps, self.data_dim])
        label_in = tf.placeholder(tf.float32, shape=[None, self.decodesteps, self.num_classes])
        noise = 0.0

        fake_enc_data, fake_enc_states, fake_dec_data, fake_dec_states = self.generator(True, data_in,
                                                                                        label_in, noise, reuse=False)
        real_enc_data, real_enc_states, real_dec_data, real_dec_states = self.generator(False, data_in,
                                                                                        label_in, noise, reuse=True)

        if flag == 'FAKE':
            autoenc_data, autoenc_label = self.autoencoder_seq2seq_pred(fake_enc_data, fake_enc_states, fake_enc_data,
                                                                        fake_enc_states, data_in, label_in, reuse=False)
        if flag == 'REAL':
            autoenc_data, autoenc_label = self.autoencoder_seq2seq_pred(real_enc_data, real_enc_states, real_enc_data,
                                                                        real_enc_states, data_in, label_in, reuse=False)

        ###################################### LOAD SAVED VARIABLES AND GENERATE DATA ##################################
        # create label dictionary
        dict_idx = np.load(data_dir + 'idx.npy')
        dict_ = dict((tuple(x.tolist()), i) for (i, x) in enumerate(dict_idx))

        print('generate data')

        init_op = tf.global_variables_initializer()
        session.run(init_op)

        saver = tf.train.Saver()

        with session.as_default():

            saver.restore(session, checkpoint)

            print('weights restored')

            for ensem in ensem_list:

                dat = np.load(data_dir + 'ensem_dat' + str(ensem) + '.npy')
                lab = np.load(data_dir + 'ensem_lab' + str(ensem) + '.npy')

                lab_ = [dict_[tuple(x.tolist())] for x in lab]
                idx = np.where(lab_ == 0)[0]

                x_train = dat[idx]
                y_train = lab[idx]
                print('ensemble:', ensem)
                syn_dat = []
                syn_lab = []

                for gen_epochs in range(self.Config.NUM_SYN_ITER):
                    print('iteration through data:', gen_epochs)

                    np.random.shuffle(x_train)

                    for i in range(int(len(x_train)) // self.batchsize):
                        data_ = x_train[i * self.batchsize: (i + 1) * self.batchsize]
                        label_ = y_train[i * self.batchsize: (i + 1) * self.batchsize]
                        syn_dat += [autoenc_data.eval(feed_dict={data_in: data_, label_in: label_})]
                        syn_lab += [autoenc_label.eval(feed_dict={data_in: data_, label_in: label_})]

                syn_dat = np.array(syn_dat)
                syn_lab = np.array(syn_lab)
                syn_lab = np.round(syn_lab)
                syn_dat.resize(syn_dat.shape[0] * syn_dat.shape[1], syn_dat.shape[2], syn_dat.shape[3])
                syn_lab.resize(syn_lab.shape[0] * syn_lab.shape[1], syn_lab.shape[2], syn_lab.shape[3])
                np.save(save_dir + 'synthetic_data_' + str(ensem) + '.npy', syn_dat)
                np.save(save_dir + 'synthetic_label_' + str(ensem) + '.npy', syn_lab)
        return

    # use this function to integrate synthetic data with real data
    def integrateSynthetic(self, ensem_dir, syn_dir, save_dir):
        # ensem_dir should only need to append lab or dat plus number
        # syn_dir should only need to append number

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        for i in range(self.Config.AE_NUM_ENSEMBLES):

            print(i)
            e_lab = np.load(ensem_dir + 'lab' + str(i) + '.npy')
            e_dat = np.load(ensem_dir + 'dat' + str(i) + '.npy')
            s_dat = np.load(syn_dir + 'synthetic_data_' + str(i) + '.npy')
            s_lab = np.load(syn_dir + 'synthetic_label_' + str(i) + '.npy')

            c_dat = np.concatenate((e_dat, s_dat), axis=0)
            c_lab = np.concatenate((e_lab, s_lab), axis=0)

            shuffle = np.random.choice(len(c_lab), len(c_lab), replace=False)

            np.save(save_dir + 'ensem_lab' + str(i) + '.npy', c_lab[shuffle])
            np.save(save_dir + 'ensem_dat' + str(i) + '.npy', c_dat[shuffle])

        return
