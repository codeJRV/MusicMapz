# TODO : Given a bunch of songs path, send it thru the model and generate predicted feature vectors here. Use this for doing TSNE

## Dummy code to create features.txt

import pickle
import tensorflow as tf
from tensorboard.plugins import projector
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
    plt.savefig(config.PLOT_PATH + "t_sne_" + input_type + '.png')

if config.LOAD_MELSPECS:
    x_data,  y_data, num_frames_test  = utils.load_h5('./datasets/saved_melspecs/all_songs.h5')
else:
    #print nb_classes
    dataset_manager.split_and_label(config.ALL_SONG_PATH,
                                    config.TRAINING_RATIO,
                                    config.TESTING_RATIO,
                                    config.VALIDATION_RATIO,
                                    config.SCALE_RATIO)
    song_paths     = utils.load('./lists/all_songs_paths.txt')
    song_labels    = utils.name2num(utils.load('./lists/all_songs_labels.txt'),tags)

    x_data, num_frames_test = melspec.extract_melgrams(song_paths, config.MULTIFRAMES, trim_song=True)
    y_data = np.array(song_labels)
    utils.save_h5('./datasets/saved_melspecs/all_songs.h5',x_data,y_data,num_frames_test)
    print('x_data shape:', x_data.shape)
    print('y_data shape:', y_data.shape)


if config.LOAD_WEIGHTS:
    y_data_categories = np_utils.to_categorical(y_data, nb_classes)
    model_path = config.WEIGHTS_PATH + "_final" + str(config.EPOCHS) + "_.h5"
    model = m.MusicTaggerCRNN(model_path, input_tensor=(1, 96, 1366), num_genres=nb_classes )
    model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
    model.summary()

    # Evaluate shape is not correct : we need to fix it from 190,1 to none, 10  --- fromh here 1) 2) incremental tsne.
    scores = model.evaluate(x_data, y_data_categories, batch_size=config.BATCH_SIZE)
    print scores

    predicted_prob = model.predict(x_data)
    print predicted_prob.shape
    print predicted_prob[0]
    predicted_classes = np.argmax(predicted_prob, axis=1)
    print predicted_classes
    print y_data
    matches=0
    for i in range(0,len(y_data)):
        if predicted_classes[i]==y_data[i]:
            matches=matches+1
    print "Accuracy %:", 100*matches/len(y_data)
    utils.save_h5(config.PLOT_PATH + "softmax_output.h5" ,predicted_prob,y_data,num_frames_test)
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

#Just dumping tsne shit

# l1 = []
# PATH = os.getcwd()
# data_path = PATH + '/data'
# data_dir_list = os.listdir(data_path)

# # GTZAN Dataset Tags
# tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
# tags = np.array(tags)

# song_data=[]
# for dataset in data_dir_list:
#     song_list=os.listdir(data_path+'/'+ dataset)
#     print ('Loaded the songs of dataset-'+'{}\n'.format(dataset))
#     for song in song_list:
#         #img1 = (imageio.imread(img).astype(np.float64)/255.0)
#         #input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
#         #im1 = cv2.resize(input_img.astype(np.float64)/255.0, (150, 150))
#         #hog1 = pyhog.features_pedro(im1, 30)


#         a = np.array()

#         ## Get and use array of features here

#         b = a.flatten()
#         l = b.tolist()

#         #print l
#         l1.append(l)

# song_features_arr = np.array(l1)
# print (song_features_arr.shape)
# np.savetxt('feature_vectors.txt',song_features_arr)


# PATH = os.getcwd()

# LOG_DIR = PATH+ '/embedding-logs'
# #metadata = os.path.join(LOG_DIR, 'metadata2.tsv')


# #%%

# feature_vectors = np.loadtxt('feature_vectors.txt')
# print ("feature_vectors_shape:",feature_vectors.shape)
# print ("num of songs:",feature_vectors.shape[0])
# print ("size of individual feature vector:",feature_vectors.shape[1])

# num_of_samples=feature_vectors.shape[0]
# print(num_of_samples)
# #num_of_samples_each_clasis = 100

# features = tf.Variable(feature_vectors, name='features')

# y = np.ones((num_of_samples,),dtype='int64')


# ### TODO : Need to use output labels of testing to apply category information

# y[0:32]=0      #texas 32
# y[32:42]=1     #stop  10
# y[42:60]=2     #streetlight 18
# y[60:89]=3      #exit 29
# y[89:210]=4      #warning 121
# y[210:235]=5      #speed 25


# ### This part depends on the output of testing


# print y


# #with open(metadata, 'w') as metadata_file:
# #    for row in range(210):
# #        c = y[row]
# #        metadata_file.write('{}\n'.format(c))
# metadata_file = open(os.path.join(LOG_DIR, 'metadata_10_classes.tsv'), 'w')
# metadata_file.write('Class\tGenre\n')

# #for i in range(210):
# #    metadata_file.write('%06d\t%s\n' % (i, names[y[i]]))
# for i in range(num_of_samples):
#     c = tags[y[i]]
#     #print(y[i], c)
#     metadata_file.write('{}\t{}\n'.format(y[i],c))
#     #metadata_file.write('%06d\t%s\n' % (j, c))
# metadata_file.close()

# with tf.Session() as sess:
#     saver = tf.train.Saver([features])

#     sess.run(features.initializer)
#     saver.save(sess, os.path.join(LOG_DIR, 'songs_10_classes.ckpt'))

#     config = projector.ProjectorConfig()
#     # One can add multiple embeddings.
#     embedding = config.embeddings.add()
#     embedding.tensor_name = features.name
#     # Link this tensor to its metadata file (e.g. labels).
#     embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_10_classes.tsv')
#     # Comment out if you don't want sprites
#     #embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_6_classes.png')
#     #embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
#     # Saves a config file that TensorBoard will read during startup.
#     projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
