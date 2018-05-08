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
ALL_SONGS_PATHS='./lists/all_songs_paths.txt'
ALL_SONGS_LABELS='./lists/all_songs_labels.txt'

TRAINING_SONGS_PATHS = "./lists/training_paths.txt"
TRAINING_SONGS_LABELS = "./lists/training_labels.txt"

TESTING_SONGS_PATHS = "./lists/testing_paths.txt"
TESTING_SONGS_LABELS = "./lists/testing_labels.txt"

VALIDATION_SONGS_PATHS = "./lists/validation_paths.txt"
VALIDATION_SONGS_LABELS = "./lists/validation_labels.txt"

## melspec files Configuration
GENRES_FILE='./lists/genre_names.txt'
ALL_SONGS_MELSPEC_FILE='./datasets/saved_melspecs/all_songs.h5'
TRAINING_MELSPEC_FILE='./datasets/saved_melspecs/training.h5'
VALIDATION_MELSPEC_FILE='./datasets/saved_melspecs/validation.h5'
TESTING_MELSPEC_FILE='./datasets/saved_melspecs/testing.h5'

### Model Configuration
SELECT_DEEP_MODELS=False
EPOCHS = 2
BATCH_SIZE = 20
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
SOFTMAX_RESULT_FILE="./Plots/softmax_output.h5"
TSNE_PLOT_PATH='./Plots/tsne_'
