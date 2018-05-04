
ALL_SONG_PATH = './datasets/all_songs/Labelled'
TRAINING_RATIO = 0.6
TESTING_RATIO = 0.2
VALIDATION_RATIO = 0.2
SCALE_RATIO = 1

MODEL_PATH  =  './saved_model/ourCRNN'
WEIGHTS_PATH = './weights/ourCRNN'

LOAD_MELSPECS = 0                       # if you dont load melspecs, then by default it means save melspecs
LOAD_MODEL    = 1                       # if you dont load model, then by default it means save model
LOAD_WEIGHTS  = 1
SAVE_WEIGHTS  = 1
PLOT_PATH = './Plots/'
# Dataset
MULTIFRAMES = 0

# Model parameters

EPOCHS = 100
BATCH_SIZE = 16
