# TODO : Training Function ( use the training set and validation set, do not touch the testing set)

import os
import time
import h5py
import sys
from keras import backend as K
from keras.optimizers import SGD
import numpy as np
from keras.utils import np_utils
from math import floor
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import melspec
import model as m
import dataset_manager

import config

import utils

# RUN ONCE ONLY AND REMEMEMBER TO SAVE MELSPECS


if config.LOAD_MELSPECS:
    x_train,  y_train, num_frames_train  = utils.load_h5('./datasets/saved_melspecs/training.h5')
    x_validate,  y_validate, num_frames_validate  = utils.load_h5('./datasets/saved_melspecs/validation.h5')
else:
    dataset_manager.split_and_label(config.ALL_SONG_PATH,
                                    config.TRAINING_RATIO,
                                    config.TESTING_RATIO,
                                    config.VALIDATION_RATIO,
                                    config.SCALE_RATIO)
    
    tags               = utils.load('./lists/genre_names.txt')
    nb_classes         = len(tags)
    print nb_classes

    training_paths     = utils.load('./lists/training_paths.txt')
    training_labels    = utils.name2num(utils.load('./lists/training_labels.txt'),tags)

    validation_paths   = utils.load('./lists/validation_paths.txt')
    validation_labels  = utils.name2num(utils.load('./lists/validation_labels.txt'),tags)

    x_train, num_frames_train = melspec.extract_melgrams(training_paths, config.MULTIFRAMES, trim_song=True)
    print('X_train shape:', x_train.shape)
    x_validate, num_frames_validate = melspec.extract_melgrams(validation_paths, config.MULTIFRAMES, trim_song=True)

    y_train    = np.array(training_labels)
    y_validate = np.array(validation_labels)

    utils.save_h5('./datasets/saved_melspecs/training.h5',x_train,y_train,num_frames_train)
    utils.save_h5('./datasets/saved_melspecs/validation.h5',x_validate,y_validate,num_frames_validate)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_validate = np_utils.to_categorical(y_validate, nb_classes)

# Initialize model

model = m.MusicTaggerCRNN(config.WEIGHTS_PATH, input_tensor=(1, 96, 1366), num_genres=nb_classes )

#model = MusicTaggerCNN(weights='msd', input_tensor=(1, 96, 1366), nb)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
















