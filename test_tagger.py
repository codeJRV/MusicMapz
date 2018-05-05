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
from matplotlib import pyplot as plt
from tsne import bh_sne
from numpy import array
import melspec
import model as m
import dataset_manager

import config

import utils

tags= utils.load('./lists/genre_names.txt')
nb_classes= len(tags)

if config.LOAD_MELSPECS:
    x_test,  y_test, num_frames_test  = utils.load_h5('./datasets/saved_melspecs/testing.h5')
else:'''
    testing_paths     = utils.load('./lists/testing_paths2.txt')
    testing_labels    = utils.name2num(utils.load('./lists/testing_labels2.txt'),tags)

    x_test, num_frames_test = melspec.extract_melgrams(testing_paths, config.MULTIFRAMES, trim_song=True)
    y_test = np.array(testing_labels)
    utils.save_h5('./datasets/saved_melspecs/testing2.h5',x_test,y_test,num_frames_test)
    print('X_test shape:', x_test.shape)
    print('Y_test shape:', y_test.shape)'''

    print "Error : No testing data to load"

if config.LOAD_WEIGHTS:
    y_test_categories = np_utils.to_categorical(y_test, nb_classes)
    model_path = config.WEIGHTS_PATH + "_final" + str(config.EPOCHS) + "_.h5"
    model = m.MusicTaggerCRNN(model_path, input_tensor=(1, 96, 1366), num_genres=nb_classes )
    model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
    model.summary()

    # Evaluate shape is not correct : we need to fix it from 190,1 to none, 10  --- fromh here 1) 2) incremental tsne.
    scores = model.evaluate(x_test, y_test_categories, batch_size=config.BATCH_SIZE)

    predicted_prob = model.predict(x_test)
    print "The test Accuracy: "
    print predicted_prob[0]
    predicted_classes = np.argmax(predicted_prob, axis=1)
    matches=0
    for i in range(0,len(y_test)):
        if predicted_classes[i]==y_test[i]:
            matches=matches+1
    print matches/len(y_test)
else:
    print 'Error: there is no model to predict'
