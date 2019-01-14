import os
import keras.backend as K
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error as mse
from utils.config import *
from utils.AttentionWithContext import AttentionWithContext


# implementation of improved wassertein gan in tensorflow

class IWGAN:

    def __init__(self, data):

        if data == 'Sentiment':
            self.Config = SentimentConfig()
        elif data == 'Power':
            self.Config = PowerConfig()
        else:
            raise ValueError('Invalid value for data option')

        self.batchsize = self.Config.GAN_BATCH_SIZE
        self.timesteps = self.Config.TIMESTEPS
        self.data_dim = self.Config.DATA_DIM
        self.data = data
        
    # define leakyrelu activation
    def leakyrelu(self, x, alpha=0.3, name='lrelu'):
        return tf.maximum(x, alpha*x)

    # define generator model
    def generator(self, noisy, real_data, noise_level, reuse=False):

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

            # stack layers of LSTM
            stacked_lstm = rnn.MultiRNNCell(
                [lstm_() for _ in range(self.Config.G_NUM_LAYERS)])

            # set initial state based on if it's real data or not
            if noisy == True:
                init_state = [lstm_state for _ in range(self.Config.G_NUM_LAYERS)]
                init_state = tuple(init_state)

            else:
                init_state = stacked_lstm.zero_state(self.batchsize, tf.float32)


            outputs, states = tf.nn.dynamic_rnn(stacked_lstm, real_data, initial_state=init_state, time_major=False,
                                                scope=scope)

            outputs = self.leakyrelu(outputs)


            if self.Config.INSTANCE_NOISE:
                output_noise = tf.random_normal([self.batchsize, self.timesteps, self.Config.G_HIDDEN_NEURONS],
                                                stddev=noise_level)
                outputs += output_noise


        return outputs, states

    # define discriminator model
    def discriminator(self, inputs, reuse=False):

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
            outputs, _, _ = rnn.static_bidirectional_rnn(stacked_lstm, stacked_lstm,
                                                         tf.unstack(tf.transpose(inputs, perm=[1, 0, 2])),
                                                         initial_state_fw=init_state, initial_state_bw=init_state,
                                                         scope=scope)

            outputs = tf.contrib.layers.fully_connected(outputs, self.Config.AE_DENSE_NEURONS, reuse=reuse, scope=scope,
                                                        activation_fn=None)
            # specify activation function here
            outputs = self.leakyrelu(outputs)
            # dropout
            outputs = tf.nn.dropout(outputs, keep_prob = self.Config.DROPOUT_OUT)
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

            # use only the last hidden state as input to dense layer
            outputs = tf.slice(outputs, [0, self.Config.TIMESTEPS - 1, 0],[self.batchsize, 1,
                                                                           self.Config.AE_DENSE_NEURONS])
            # use last hidden state as input to a dense layer
            outputs = tf.layers.dense(outputs, int(self.Config.AE_DENSE_NEURONS/2), name='discriminator/pre_output')
            # specify activation function here
            outputs = self.leakyrelu(outputs)
            # dropout
            outputs = tf.nn.dropout(outputs, keep_prob=self.Config.DROPOUT_OUT)
            # final output of discriminator
            outputs = tf.layers.dense(outputs, 1, name='discriminator/output')

        return outputs

    # build autoencoder seq2seq model with predictions
    def autoencoder_seq2seq_pred(self, inputs, states, data_in, reuse=False):

        with tf.variable_scope('autoencoder', reuse=reuse) as scope:
            # get attention from hidden states of generator
            atten = AttentionWithContext()(inputs)

            states = tf.unstack(states, axis=0)
            # set inital cell state to cell state from generator
            # set inital hidden state to attention of hidden states from generator
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(states[idx][0], atten) for idx in range(self.Config.A_NUM_LAYERS)])

            def lstm_():
                return rnn.DropoutWrapper(rnn.LSTMCell(self.Config.A_HIDDEN_NEURONS),
                                          input_keep_prob=self.Config.DROPOUT_IN, output_keep_prob=self.Config.DROPOUT_OUT)

            stacked_lstm = rnn.MultiRNNCell(
                [lstm_() for _ in range(self.Config.G_NUM_LAYERS)])

            ae_in = tf.zeros([self.batchsize, 1, self.Config.DATA_DIM])

            all_outputs = []

            # feed in previous output as current input (teacher forcing)
            for j in range(self.Config.TIMESTEPS):
                outputs, states = tf.nn.dynamic_rnn(stacked_lstm, ae_in, initial_state=rnn_tuple_state,
                                                    time_major=False,
                                                    scope=scope)
                outputs = self.leakyrelu(outputs)
                outputs = tf.layers.dense(outputs, self.Config.DATA_DIM, name='autoencoder/output')

                all_outputs += [outputs]
                ae_in = tf.expand_dims(data_in[:, j, :], axis=1)

                scope.reuse_variables()

            outputs = tf.concat(all_outputs, 1)

        return outputs

    def iwganAeTrain(self, x_train, save_dir):

        d_loss = []
        d_loss_diff = []
        g_loss = []
        a_loss = []
        g_loss_diff = []

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_in = tf.placeholder(tf.float32, shape=[None, self.timesteps, self.data_dim])
        noise = tf.placeholder(tf.float32, shape=())

        fake_data, fake_states = self.generator(True, data_in, noise, reuse=False)
        real_data, real_states = self.generator(False, data_in, noise, reuse=True)
        disc_fake = self.discriminator(fake_data, reuse=False)
        disc_real = self.discriminator(real_data, reuse=True)

        autoenc_real = self.autoencoder_seq2seq_pred(real_data, real_states, data_in, reuse=False)

        # calculate loss
        ae_cost = tf.reduce_mean(tf.norm(tf.reshape(data_in, [self.batchsize, self.timesteps*self.data_dim])-
                                         tf.reshape(autoenc_real,[self.batchsize, self.timesteps*self.data_dim]),
                                         axis=1))

        disc_cost_ = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        gen_cost_ = -tf.reduce_mean(disc_fake)
        gen_cost = gen_cost_ + self.Config.MU*ae_cost

        # gradient loss
        alpha = tf.random_uniform(
            shape=[self.batchsize, 1, 1],
            minval=0.,
            maxval=1.
        )
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        disc_interpolates = self.discriminator(interpolates, reuse=True)
        gradients = tf.gradients(disc_interpolates, [interpolates])[0]
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
                minibatch_size = self.batchsize * (self.Config.DISC_CRITIC_ITERS+self.Config.GEN_CRITIC_ITERS+1)

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
                    print('minibatch:', i)
                    for j in range(self.Config.GEN_CRITIC_ITERS):
                        _data = data_minibatch[j * self.batchsize: (j + 1) * self.batchsize]
                        _gen_cost, _gen_cost_diff,  _ = session.run([gen_cost, gen_cost_, gen_train_op],
                                                                    feed_dict={data_in: _data, noise: _noise})
                        g_loss += [_gen_cost]
                        g_loss_diff += [_gen_cost_diff]
                    for j in range(self.Config.DISC_CRITIC_ITERS):
                        _data = data_minibatch[(self.Config.GEN_CRITIC_ITERS+j)*self.batchsize:
                        (self.Config.GEN_CRITIC_ITERS + j+1) * self.batchsize]
                        _disc_cost, _disc_cost_diff, _ = session.run([disc_cost, disc_cost_, disc_train_op],
                                                                     feed_dict={data_in: _data, noise: _noise})
                        d_loss += [_disc_cost]
                        d_loss_diff += [_disc_cost_diff]

                    _data = data_minibatch[(self.Config.DISC_CRITIC_ITERS+self.Config.GEN_CRITIC_ITERS) *
                                           self.batchsize: (self.Config.GEN_CRITIC_ITERS +
                                                            self.Config.DISC_CRITIC_ITERS+1) * self.batchsize]
                    _ae_cost, _ = session.run([ae_cost, autoenc_train_op], feed_dict={data_in: _data, noise: _noise})
                    a_loss += [_ae_cost]

                if epoch >= 0 and epoch % self.Config.CHECKPOINT_STEP == 0:
                    saver.save(session, save_dir + 'model', global_step=epoch)
                    np.save(save_dir + 'd_loss.npy', d_loss)
                    np.save(save_dir + 'g_loss.npy', g_loss)
                    np.save(save_dir + 'g_diff_loss.npy', g_loss_diff)
                    np.save(save_dir + 'd_diff_loss.npy', d_loss_diff)
                    np.save(save_dir + 'a_loss.npy', a_loss)

            saver.save(session, save_dir + 'model', global_step=self.Config.GAN_AE_EPOCHS-1)
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
        noise = 0.0

        fake_data, fake_states = self.generator(True, data_in, noise, reuse=False)
        real_data, real_states = self.generator(False, data_in, noise, reuse=True)

        if flag == 'FAKE':
            autoenc_fake = self.autoencoder_seq2seq_pred(fake_data, fake_states, data_in, reuse=False)
        elif flag == 'REAL':
            autoenc_fake = self.autoencoder_seq2seq_pred(real_data, real_states, data_in, reuse=False)
        ###################################### LOAD SAVED VARIABLES AND GENERATE DATA ##################################

        print('generate data')

        init_op = tf.global_variables_initializer()
        session.run(init_op)

        saver = tf.train.Saver()

        with session.as_default():

            saver.restore(session, checkpoint)

            print('weights restored')

            for ensem in range(self.Config.AE_NUM_ENSEMBLES):

                dat = np.load(data_dir+'ensem_dat'+str(ensem)+'.npy')
                lab = np.load(data_dir+'ensem_lab'+str(ensem)+'.npy')

                if len(lab.shape) == 3:
                    idx = np.where(lab[:, -1, 1] == 1)[0]
                else:
                    idx = np.where(lab[:, 1] == 1)[0]
                x_train = dat[idx]
                print('ensemble:', ensem)
                syn_dat = []

                for gen_epochs in range(self.Config.NUM_SYN_ITER):
                    print('iteration through data:', gen_epochs)

                    np.random.shuffle(x_train)

                    for i in range(int(len(x_train)) // self.batchsize):
                        data_ = x_train[i * self.batchsize: (i + 1) * self.batchsize]
                        syn_dat += [autoenc_fake.eval(feed_dict={data_in: data_})]

                syn_dat = np.array(syn_dat)
                syn_dat.resize(syn_dat.shape[0] * syn_dat.shape[1], syn_dat.shape[2], syn_dat.shape[3])
                np.save(save_dir + 'synthetic_data_' + str(ensem) + '.npy', syn_dat)
        return

    # do iwgan accuracy
    def iwganAcc(self, save_dir, x_train, checkpoint):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        session = tf.Session()
        K.set_session(session)

        # get sizes of batches
        num_batch = int(len(x_train)//self.batchsize)
        test_size = int(num_batch/2)

        # shuffle dataset
        np.random.shuffle(x_train)

        print('calculate loss function')
        # get model outputs
        data_in = tf.placeholder(tf.float32, shape=[None, self.timesteps, self.data_dim])
        noise = 0.0

        fake_data, fake_states = self.generator(True, data_in, noise, reuse=False)
        real_data, real_states = self.generator(False, data_in, noise, reuse=True)
        disc_fake = self.discriminator(fake_data, reuse=False)
        disc_real = self.discriminator(real_data, reuse=True)

    ###################################### LOAD SAVED VARIABLES AND GENERATE DATA ######################################

        print('generate data')

        init_op = tf.global_variables_initializer()
        session.run(init_op)

        saver = tf.train.Saver()

        with session.as_default():
            saver.restore(session, checkpoint)

            real_disc_lab = []
            fake_disc_lab = []
            for i in range(test_size):
                data_ = x_train[i * self.batchsize: (i + 1) * self.batchsize]
                real_disc_lab += [disc_real.eval(feed_dict={data_in: data_})]
                fake_disc_lab += [disc_fake.eval(feed_dict={data_in: data_})]

            real_disc_lab = np.array(real_disc_lab)
            fake_disc_lab = np.array(fake_disc_lab)
            real_disc_lab.resize(real_disc_lab.shape[0] * real_disc_lab.shape[1], real_disc_lab.shape[2])
            fake_disc_lab.resize(fake_disc_lab.shape[0] * fake_disc_lab.shape[1], fake_disc_lab.shape[2])

            print('try tau values')
            # now try tau
            tau_ = np.array([x / 10 for x in range(-30, 10)])  # np.array([x/20 for x in range(1,11)])
            tau_f1 = []
            real_lab = np.zeros(real_disc_lab.shape)
            fake_lab = np.ones(fake_disc_lab.shape)

            lab = np.concatenate((real_lab, fake_lab))

            for tau in tau_:
                val_real_lab = np.where(real_disc_lab > tau, 1, 0)
                val_fake_lab = np.where(fake_disc_lab > tau, 1, 0)
                val_lab = np.concatenate((val_real_lab, val_fake_lab))
                tau_f1 += [f1_score(lab, val_lab)]
            tau = tau_[np.argmax(tau_f1)]
            val_f1 = max(tau_f1)

            # test on data now
            real_disc_lab = []
            fake_disc_lab = []
            for i in range(test_size, 2 * test_size):
                data_ = x_train[i * self.batchsize: (i + 1) * self.batchsize]
                real_disc_lab += [disc_real.eval(feed_dict={data_in: data_})]
                fake_disc_lab += [disc_fake.eval(feed_dict={data_in: data_})]

            real_disc_lab = np.array(real_disc_lab)
            fake_disc_lab = np.array(fake_disc_lab)
            real_disc_lab.resize(real_disc_lab.shape[0] * real_disc_lab.shape[1], real_disc_lab.shape[2])
            fake_disc_lab.resize(fake_disc_lab.shape[0] * fake_disc_lab.shape[1], fake_disc_lab.shape[2])

            print('test on test data')

            real_lab = np.zeros(real_disc_lab.shape)
            fake_lab = np.ones(fake_disc_lab.shape)
            lab = np.concatenate((real_lab, fake_lab))

            test_real_lab = np.where(real_disc_lab > tau, 1, 0)
            test_fake_lab = np.where(fake_disc_lab > tau, 1, 0)
            test_lab = np.concatenate((test_real_lab, test_fake_lab))

            print('calculate test f1')
            test_f1 = [f1_score(lab, test_lab)]

            np.save(save_dir + 'test_f1.npy', test_f1)
            np.save(save_dir + 'val_f1.npy', val_f1)
            np.save(save_dir + 'tau.npy', tau)

        return

    # use this to get model accuracy based on the discriminator model
    def iwganDisc(self, save_dir, x_val, y_val, x_test, y_test, checkpoint):

        session = tf.Session()
        K.set_session(session)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if len(y_val.shape) == 3:
            y_val = y_val[:, -1, :]
        y_val = y_val.argmax(axis=1)

        if len(y_test.shape) == 3:
            y_test = y_test[:, -1, :]
        y_test = y_test.argmax(axis=1)

        print('calculate loss function')
        # get model outputs
        data_in = tf.placeholder(tf.float32, shape=[None, self.timesteps, self.data_dim])
        noise = 0.0

        real_data, real_states = self.generator(False, data_in, noise, reuse=True)
        disc_real = self.discriminator(real_data, reuse=True)

        ###################################### LOAD SAVED VARIABLES AND GENERATE DATA ##################################
        print('generate data')

        init_op = tf.global_variables_initializer()
        session.run(init_op)

        saver = tf.train.Saver()

        with session.as_default():
            saver.restore(session, checkpoint)

            idx = np.array(range(len(x_val)))
            idx_ = np.array(range(len(x_test)))

            np.random.shuffle(idx)
            np.random.shuffle(idx_)

            x_val = x_val[idx]
            y_val = y_val[idx]

            x_test = x_test[idx_]
            y_test = y_test[idx_]

            disc_lab = []
            for i in range(int(len(y_val) // self.batchsize)):
                data_ = x_val[i * self.batchsize: (i + 1) * self.batchsize]
                disc_lab += [disc_real.eval(feed_dict={data_in: data_})]

            disc_lab = np.array(disc_lab)
            disc_lab.resize(disc_lab.shape[0] * disc_lab.shape[1], disc_lab.shape[2])

            print(disc_lab.shape)
            print(y_val[0:int(len(y_val) // self.batchsize) * self.batchsize].shape)

            print('try tau values')
            # now try tau
            tau_ = np.array([x / 10 for x in range(-30, 10)])  # np.array([x/20 for x in range(1,11)])
            tau_f1 = []
            for tau in tau_:
                val_test_lab = np.where(disc_lab > tau, 1, 0)
                tau_f1 += [f1_score(y_val[0:int(len(y_val) // self.batchsize) * self.batchsize], val_test_lab)]

            tau = tau_[np.argmax(tau_f1)]
            val_f1 = max(tau_f1)

            disc_lab = []
            print('predict on test data')
            # evaluate the model on the validation data
            for i in range(int(len(x_test) // self.batchsize)):
                data_ = x_test[i * self.batchsize: (i + 1) * self.batchsize]
                disc_lab += [disc_real.eval(feed_dict={data_in: data_})]

            disc_lab = np.array(disc_lab)
            disc_lab.resize(disc_lab.shape[0] * disc_lab.shape[1], disc_lab.shape[2])

            test_pred_lab = np.where(disc_lab > tau, 1, 0)
            print(test_pred_lab.shape)
            print(y_test[0:int(len(y_test) // self.batchsize) * self.batchsize].shape)

            print('calculate test f1')
            test_f1 = f1_score(y_test[0:int(len(x_test) // self.batchsize) * self.batchsize], test_pred_lab)

            np.save(save_dir + 'test_f1.npy', test_f1)
            np.save(save_dir + 'val_f1.npy', val_f1)
            np.save(save_dir + 'tau.npy', tau)
        return

    # use this to get novelty detection
    def iwganNovelty(self, save_dir, x_val, y_val, x_test, y_test, checkpoint):
        session = tf.Session()
        K.set_session(session)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if len(y_val.shape) == 3:
            y_val = y_val[:, -1, :]
        y_val = y_val.argmax(axis=1)

        if len(y_test.shape) == 3:
            y_test = y_test[:, -1, :]
        y_test = y_test.argmax(axis=1)

        # get model outputs
        data_in = tf.placeholder(tf.float32, shape=[None, self.timesteps, self.data_dim])
        noise = 0.0

        real_data, real_states = self.generator(False, data_in, noise, reuse=True)

        autoenc_real = self.autoencoder_seq2seq_pred(real_data, real_states, data_in, reuse=False)
        ###################################### LOAD SAVED VARIABLES AND GENERATE DATA ##################################

        print('generate data')

        init_op = tf.global_variables_initializer()
        session.run(init_op)

        saver = tf.train.Saver()
        syn_dat = []

        with session.as_default():
            saver.restore(session, checkpoint)

            idx = np.array(range(len(x_val)))
            idx_ = np.array(range(len(x_test)))

            np.random.shuffle(idx)
            np.random.shuffle(idx_)

            x_test = x_test[idx_]
            y_test = y_test[idx_]

            x_val = x_val[idx]
            y_val = y_val[idx]

            for i in range(int(len(y_val) // self.batchsize)):
                data_ = x_val[i * self.batchsize: (i + 1) * self.batchsize]
                syn_dat += [autoenc_real.eval(feed_dict={data_in: data_})]

            syn_dat = np.array(syn_dat)
            syn_dat.resize(syn_dat.shape[0] * syn_dat.shape[1], syn_dat.shape[2] * syn_dat.shape[3])
            x_val.resize(x_val.shape[0], x_val.shape[1] * x_val.shape[2])
            # now compare the val_dat and autoencoder loss

            loss = []

            print('calculate loss')
            for i in range(int(len(x_val) // self.batchsize) * self.batchsize):
                loss += [mse(x_val[i], syn_dat[i])]

            print('try tau values')
            # now try tau
            tau_ = np.array([x / 20 for x in range(1, 41)])  # np.array([x/20 for x in range(1,11)])
            tau_f1 = []
            for tau in tau_:
                val_test_lab = np.where(loss > tau, 1, 0)
                tau_f1 += [f1_score(y_val[0:int(len(y_val) // self.batchsize) * self.batchsize], val_test_lab)]

            tau = tau_[np.argmax(tau_f1)]
            val_f1 = max(tau_f1)

            syn_dat = []
            print('predict on test data')
            # evaluate the model on the validation data
            for i in range(int(len(x_test) // self.batchsize)):
                data_ = x_test[i * self.batchsize: (i + 1) * self.batchsize]
                syn_dat += [autoenc_real.eval(feed_dict={data_in: data_})]

            syn_dat = np.array(syn_dat)
            syn_dat.resize(syn_dat.shape[0] * syn_dat.shape[1], syn_dat.shape[2] * syn_dat.shape[3])

            # calculate loss for each of the model
            x_test.resize(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

            print('calculate loss')
            loss = []
            for i in range(int(len(x_test) // self.batchsize) * self.batchsize):
                loss += [mse(x_test[i], syn_dat[i])]

            test_pred_lab = np.where(loss > tau, 1, 0)

            print('calculate test f1')
            test_f1 = f1_score(y_test[0:int(len(y_test) // self.batchsize) * self.batchsize], test_pred_lab)

            np.save(save_dir + 'test_f1.npy', test_f1)
            np.save(save_dir + 'val_f1.npy', val_f1)
            np.save(save_dir + 'tau.npy', tau)

        return

    # use this function to integrate synthetic data with real data
    def integrateSynthetic(self, ensem_dir, syn_dir, save_dir, y_min):
        # ensem_dir should only need to append lab or dat plus number
        # syn_dir should only need to append number

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        minority_label = y_min[0]

        for i in range(self.Config.AE_NUM_ENSEMBLES):

            print(i)
            e_lab = np.load(ensem_dir + 'lab' + str(i) + '.npy')
            e_dat = np.load(ensem_dir + 'dat' + str(i) + '.npy')
            s_dat = np.load(syn_dir + str(i) + '.npy')

            # build synthetic label
            s_lab = []
            for j in range(len(s_dat)):
                s_lab += [minority_label]
            s_lab = np.array(s_lab)

            c_dat = np.concatenate((e_dat, s_dat), axis=0)
            c_lab = np.concatenate((e_lab, s_lab), axis=0)

            shuffle = np.random.choice(len(c_lab), len(c_lab), replace=False)

            np.save(save_dir + 'ensem_lab' + str(i) + '.npy', c_lab[shuffle])
            np.save(save_dir + 'ensem_dat' + str(i) + '.npy', c_dat[shuffle])

        return
