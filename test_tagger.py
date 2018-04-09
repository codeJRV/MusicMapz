# TODO test tagger performance function

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
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt

import melspec
import model as m
import dataset_manager

import config

import utils

if config.LOAD_MELSPECS:
    x_test,  y_test, num_frames_test  = utils.load_h5('./datasets/saved_melspecs/testing.h5')
else:

    tags               = utils.load('./lists/genre_names.txt')
    nb_classes         = len(tags)
    print nb_classes

    testing_paths     = utils.load('./lists/testing_paths.txt')
    testing_labels    = utils.name2num(utils.load('./lists/testing_labels.txt'),tags)
    
    x_test, num_frames_train = melspec.extract_melgrams(testing_paths, config.MULTIFRAMES, trim_song=True)
    y_test = np.array(testing_labels)
     
    print('X_test shape:', x_test.shape)
    print('Y_test shape:', y_test.shape)



    if config.LOAD_WEIGHTS:
        model_path = config.WEIGHTS_PATH + "_final_.h5"
        model = m.MusicTaggerCRNN(config.WEIGHTS_PATH, input_tensor=(1, 96, 1366), num_genres=nb_classes )

        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        model.summary()

        scores = model.evaluate(x_test, y_test, batch_size=config.BATCH_SIZE)
        print('mse=%f, mae=%f, mape=%f' % (scores[0],scores[1],scores[2]))

        #Perfrom TSNE using scikit learn
        random_seed  = 0
        weights = model.get_layer('Flatten_1').get_weights()
        tsne = TSNE(n_components=2, random_state=random_seed, verbose=1)
        transformed_weights = tsne.fit_transform(weights)

    else:
        print 'there is no model to predict'



