import os
from random import shuffle
import math
import json
from random import shuffle
import math
from sklearn.model_selection import train_test_split
from dataset_manager import get_all_song_paths_and_labels, get_all_song_paths_and_labels_FMA

def split_and_label():
    with open(config.ALL_SONGS_LABELS) as f:
        genres = f.read().splitlines()

    with open(config.ALL_SONGS_PATHS) as g:
        songs = g.read().splitlines()

    X_train, X_test, y_train, y_test = train_test_split(songs, genres, test_size=config.TESTING_RATIO, random_state=1)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=config.VALIDATION_RATIO, random_state=1)

    with open (training_path,"w")as fp:
        for line in X_train:
            fp.write(line+"\n")
    with open (training_label,"w")as fp:
        for line in y_train:
            fp.write(line+"\n")

    with open (testing_path,"w")as fp:
        for line in X_test:
            fp.write(line+"\n")
    with open (testing_label,"w")as fp:
        for line in y_test:
            fp.write(line+"\n")


    with open (validation_path,"w")as fp:
        for line in X_val:
            fp.write(line+"\n")
    with open (validation_label,"w")as fp:
        for line in y_val:
            fp.write(line+"\n")

    print (len(genres))
    print ( len (X_train))
    print ( len (X_test))
    print (len(X_val))

    allsongdict = {}

    for genre in genres:
        if not (genre in allsongdict):
            allsongdict[genre] = []

    for i in range(len(genres)):
        allsongdict[genres[i]].append(songs[i])

    for key in allsongdict:
        print ( key, ':', len(allsongdict[key]))
        shuffle(allsongdict[key])

def generate_h5_files():
        training_paths     = utils.load(config.TRAINING_SONGS_PATHS)
        training_labels    = utils.name2num(utils.load(config.TRAINING_SONGS_LABELS),tags)

        validation_paths   = utils.load(config.VALIDATION_SONGS_PATHS)
        validation_labels  = utils.name2num(utils.load(config.VALIDATION_SONGS_LABELS),tags)

        testing_paths     = utils.load(config.TESTING_SONGS_PATHS)
        testing_labels    = utils.name2num(utils.load(config.TESTING_SONGS_LABELS),tags)

        x_test, num_frames_test = melspec.extract_melgrams(testing_paths, config.MULTIFRAMES, trim_song=True)
        x_train, num_frames_train = melspec.extract_melgrams(training_paths, config.MULTIFRAMES, trim_song=True)
        x_validate, num_frames_validate = melspec.extract_melgrams(validation_paths, config.MULTIFRAMES, trim_song=True)

        y_train    = np.array(training_labels)
        y_validate = np.array(validation_labels)
        y_test = np.array(testing_labels)

        utils.save_h5(config.TRAINING_MELSPEC_FILE,x_train,y_train,num_frames_train)
        utils.save_h5(config.VALIDATION_MELSPEC_FILE,x_validate,y_validate,num_frames_validate)
        utils.save_h5(config.VALIDATION_MELSPEC_FILE,x_test,y_test,num_frames_test)

get_all_song_paths_and_labels()
split_and_label()
generate_h5_files()
