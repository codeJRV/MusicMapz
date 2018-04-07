
ALL_SONG_PATH = './datasets/all_songs/Labelled'
TRAINING_RATIO = 0.6
TESTING_RATIO = 0.2
VALIDATION_RATIO = 0.2
SCALE_RATIO = 0.1

MODEL_PATH  =  './saved_model/ourCRNN'
WEIGHTS_PATH = './saved_weights/ourCRNN'

LOAD_MELSPECS = 0                       # if you dont load melspecs, then by default it means save melspecs
LOAD_MODEL    = 0                       # if you dont load model, then by default it means save model
LOAD_WEIGHTS  = 0                       # if you dont load weights, then by default it means save weights

# Dataset
MULTIFRAMES = 0

# Model parameters

epochs = 40
batch_size = 100
time_elapsed = 0
