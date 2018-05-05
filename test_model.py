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

import dataset_manager
import config
import utils

if config.SELECT_DEEP_MODELS:
    import model_deep as m
else:
    import model as m

tags= utils.load(config.GENRES_FILE_PATH)
nb_classes= len(tags)

if config.LOAD_MELSPECS:
    x_test,  y_test, num_frames_test  = utils.load_h5(config.TESTING_MELSPEC_FILE)
else:
    print "Error : No testing data to load"
    sys.exit()

if config.LOAD_WEIGHTS:
    y_test_categories = np_utils.to_categorical(y_test, nb_classes)
    model_path =
    model = m.MusicTaggerCRNN(model_path, input_tensor=(1, 96, 1366), num_genres=nb_classes )
    model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
    model.summary()

    # Evaluate shape is not correct : we need to fix it from 190,1 to none, 10  --- fromh here 1) 2) incremental tsne.
    scores = model.evaluate(x_test, y_test_categories, batch_size=config.BATCH_SIZE)
    print('Test Loss:', scores[0])
    print('Test Accuracy:', 100*scores[1])
else:
    print 'Error: there is no model to predict'
