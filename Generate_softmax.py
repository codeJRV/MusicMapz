# TODO : Given a bunch of songs path, send it thru the model and generate predicted feature vectors here. Use this for doing TSNE

## Dummy code to create features.txt

import os
import time
import h5py
import sys
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from math import floor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from tsne import bh_sne
from numpy import array
import config
if config.SELECT_DEEP_MODELS:
    import model_deep as m
else:
    import model as m
#import dataset_manager
import config
from preprocess_songs import merge_all_h5s
import utils

tags= utils.load(config.GENRES_FILE)
nb_classes= len(tags)

def plot_tsne(x_data,y_data,input_type):
    fig = plt.figure()
    reshapedList = array(x_data)
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
    plot_title="t-SNE for " +  input_type + "samples as genres"
    plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10), label='Genres')
    plt.clim(-0.5, 9.5)
    plt.title(plot_title)
    plt.savefig(config.TSNE_PLOT_PATH + input_type + '.png')

## Code starts here



if config.LOAD_MELSPECS:
    x_data,  y_data, num_frames_test  = utils.load_h5(config.ALL_SONGS_MELSPEC_FILE)
    print x_data.shape
    print y_data.shape
else:
    #print "Error: No input to load"
    #sys.exit()
    merge_all_h5s()
    x_data,  y_data, num_frames_test  = utils.load_h5(config.ALL_SONGS_MELSPEC_FILE)
    print x_data.shape
    print y_data.shape

if config.LOAD_WEIGHTS:
    y_data_categories = np_utils.to_categorical(y_data, nb_classes)
    model = m.MusicTaggerCRNN(config.MODEL_WEIGHTS_FILE, input_tensor=(1, 96, 1366), num_genres=nb_classes )
    model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
    model.summary()

    # Evaluate shape is not correct : we need to fix it from 190,1 to none, 10  --- fromh here 1) 2) incremental tsne.
    scores = model.evaluate(x_data, y_data_categories, batch_size=config.BATCH_SIZE)
    #print scores

    predicted_prob = model.predict(x_data)
    #print predicted_prob.shape
    #print predicted_prob[0]
    predicted_classes = np.argmax(predicted_prob, axis=1)
    print predicted_classes
    print y_data
    matches=0
    for i in range(0,len(y_data)):
        if predicted_classes[i]==y_data[i]:
            matches=matches+1
    print "Accuracy %:", 100*matches/len(y_data)
    utils.save_h5(config.SOFTMAX_RESULT_FILE,predicted_prob,y_data,num_frames_test)
    plot_tsne(x_data,y_data,"melspectrogram")
    plot_tsne(predicted_prob,y_data,"CRNN_features")

    #print('mse=%f, mae=%f, mape=%f' % (scores[0],scores[1],scores[2]))
    #Perfrom TSNE using scikit learn
    '''random_seed  = 0
    weights = model.get_layer('Flatten_1').get_weights()
    tsne = TSNE(n_components=2, random_state=random_seed, verbose=1)
    transformed_weights = tsne.fit_transform(weights)'''
else:
    print 'there is no model to predict'
