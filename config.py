## Data split configuration
SONG_FLODER = './datasets/all_songs/Labelled'
TRAINING_RATIO = 0.6
TESTING_RATIO = 0.2
VALIDATION_RATIO = 0.2
SCALE_RATIO = 1

## FMA Folder
SONG_FLODER_FMA='/media/jrv/Data/fma_large'
FMA_DATASET_CSV='/home/jrv/Desktop/jrv1/fma/tracks.csv'
## RAW FILE PATHS and labels
ALL_SONGS_PATHS='./lists/all_songs_paths_fma.txt'
ALL_SONGS_LABELS='/lists/all_songs_labels_fma.txt'

TRAINING_SONGS_PATHS = "./lists/training_paths_fma.txt"
TRAINING_SONGS_LABELS = "./lists/training_labels_fma.txt"

TESTING_SONGS_PATHS = "./lists/testing_paths_fma.txt"
TESTING_SONGS_LABELS = "./lists/testing_labels_fma.txt"

VALIDATION_SONGS_PATHS = "./lists/validation_paths_fma.txt"
VALIDATION_SONGS_LABELS = "./lists/validation_labels_fma.txt"

## melspec files Configuration
GENRES_FILE='./lists/genre_names.txt'
ALL_SONGS_MELSPEC_FILE='./datasets/saved_melspecs/all_songs_fma.h5'
TRAINING_MELSPEC_FILE='./datasets/saved_melspecs/training_fma.h5'
VALIDATION_MELSPEC_FILE='./datasets/saved_melspecs/validation_fma.h5'
TESTING_MELSPEC_FILE='./datasets/saved_melspecs/testing_fma.h5'

### Model Configuration
SELECT_DEEP_MODELS=True
EPOCHS = 100
BATCH_SIZE = 16
MODEL_PATH  =  './saved_model/ourCRNN'
MODEL_WEIGHTS_FILE = "./weights/ourCRNN_final" + str(EPOCHS) + "_.h5"

LOAD_MELSPECS = 1                       # if you dont load melspecs, then by default it means save melspecs
LOAD_MODEL    = 1                       # if you dont load model, then by default it means save model
LOAD_WEIGHTS  = 1
SAVE_WEIGHTS  = 1
PLOT_PATH = './Plots/'
# Dataset
MULTIFRAMES = 0

## tSNE- and model softmax layer
SOFTMAX_RESULT_FILE="./Plots/softmax_output_fma.h5"
TSNE_PLOT_PATH='./Plots/tsne_'
