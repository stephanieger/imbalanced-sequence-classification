## Configuration file for Power dataset
class PowerConfig:

    # parameters for preprocessing data
    PAD = False

    # parameters for data
    TIMESTEPS = 20
    DECODESTEPS = 4
    DATA_DIM = 6
    NUM_CLASSES = 2

    # model building parameters
    HIDDEN_NEURONS = 8
    DENSE_HIDDEN_NEURONS = 8
    NUM_LAYERS = 5

    # model training parameters
    BATCH_SIZE = 256
    GAN_BATCH_SIZE = 1024
    EPOCHS = 100
    NUM_ENSEMBLES = 10

    # smote model parameters
    N = 1 # number of times to resample each point
    K = 5 #
    BETA = 0.001

    # autoencoder parameters
    # model building parameters
    AE_EPOCHS = 200
    GAN_AE_EPOCHS = 3000
    G_HIDDEN_NEURONS = 64
    G_NUM_LAYERS = 2
    D_HIDDEN_NEURONS = 64
    D_NUM_LAYERS = 2
    A_HIDDEN_NEURONS = 64
    A_NUM_LAYERS=2
    AE_DENSE_NEURONS = 32
    DROPOUT_IN = 0.8
    DROPOUT_OUT = 0.8
    DROPOUT_STATE = 1
    INSTANCE_NOISE = True

    # hyperparameters
    LAMBDA = 10
    MU = 1

    # training parameters
    GEN_CRITIC_ITERS = 2
    DISC_CRITIC_ITERS = 1
    LEARNING_RATE = 1e-4
    DISC_LEARNING_RATE = 1e-5
    EP = 1e-10
    CHECKPOINT_STEP = 100

    # parameters for generating data
    NUM_SYN_ITER = 3
    LAST_EPOCH = GAN_AE_EPOCHS-1
    AE_NUM_ENSEMBLES = 10


class SentimentConfig:

    # parameters for preprocessing data
    PAD = True

    # parameters for data
    TIMESTEPS = 600
    DATA_DIM = 300
    NUM_CLASSES = 2

    # model building parameters
    HIDDEN_NEURONS = 64
    DENSE_HIDDEN_NEURONS = 64
    NUM_LAYERS = 3

    # model training parameters
    BATCH_SIZE = 128
    GAN_BATCH_SIZE = 64
    EPOCHS = 100
    NUM_ENSEMBLES = 10

    # smote model parameters
    N = 1 # number of times to resample each point
    K = 5 #
    BETA = 0.001

    # autoencoder parameters
    # model building parameters
    AE_EPOCHS = 200
    GAN_AE_EPOCHS = 2500
    G_HIDDEN_NEURONS = 64
    G_NUM_LAYERS = 2
    D_HIDDEN_NEURONS = 64
    D_NUM_LAYERS = 2
    A_HIDDEN_NEURONS = 64
    A_NUM_LAYERS=2
    AE_DENSE_NEURONS = 32
    DROPOUT_IN = 0.8
    DROPOUT_OUT = 0.8
    DROPOUT_STATE = 1
    INSTANCE_NOISE = True

    # hyperparameters
    LAMBDA = 10
    MU = 0.1

    # training parameters
    GEN_CRITIC_ITERS = 1
    DISC_CRITIC_ITERS = 2
    LEARNING_RATE = 1e-3
    DISC_LEARNING_RATE = 1e-3
    EP = 1e-10
    CHECKPOINT_STEP = 100

    # parameters for generating data
    NUM_SYN_ITER = 3
    LAST_EPOCH = GAN_AE_EPOCHS-1
    AE_NUM_ENSEMBLES = 10
