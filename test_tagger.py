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
else:

    #print nb_classes
    testing_paths     = utils.load('./lists/testing_paths.txt')
    testing_labels    = utils.name2num(utils.load('./lists/testing_labels.txt'),tags)

    x_test, num_frames_test = melspec.extract_melgrams(testing_paths, config.MULTIFRAMES, trim_song=True)
    y_test = np.array(testing_labels)
    utils.save_h5('./datasets/saved_melspecs/testing.h5',x_test,y_test,num_frames_test)
    print('X_test shape:', x_test.shape)
    print('Y_test shape:', y_test.shape)


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
    print scores

    predicted_prob = model.predict(x_test)
    print predicted_prob.shape
    print predicted_prob[0]
    predicted_classes = np.argmax(predicted_prob, axis=1)
    print predicted_classes
    print y_test
    matches=0
    for i in range(0,len(y_test)):
        if predicted_classes[i]==y_test[i]:
            matches=matches+1
    print matches/len(y_test)

    reshapedList = array(predicted_prob)
    x_data = np.asarray(reshapedList).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    # perform t-SNE embedding
    vis_data = bh_sne(x_data, perplexity=30)

    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    '''colors = []
    for label in labels:
        colors.append(label.index(1))'''

    plt.scatter(vis_x, vis_y, c=y_test, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10), label='Genres')
    plt.clim(-0.5, 9.5)
    plt.title('t-SNE mel-spectrogram samples as genres')
    plt.show()

    #print('mse=%f, mae=%f, mape=%f' % (scores[0],scores[1],scores[2]))
    #Perfrom TSNE using scikit learn
    '''random_seed  = 0
    weights = model.get_layer('Flatten_1').get_weights()
    tsne = TSNE(n_components=2, random_state=random_seed, verbose=1)
    transformed_weights = tsne.fit_transform(weights)'''
else:
    print 'there is no model to predict'
