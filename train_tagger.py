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
# import dataset_manager

import config

import utils

# RUN ONCE ONLY AND REMEMEMBER TO SAVE MELSPECS
tags               = utils.load('./lists/genre_names.txt')
nb_classes         = len(tags)
print (nb_classes)

if config.LOAD_MELSPECS:
    x_train,  y_train, num_frames_train  = utils.load_h5('./datasets/saved_melspecs/training2.h5')
    x_validate,  y_validate, num_frames_validate  = utils.load_h5('./datasets/saved_melspecs/validation2.h5')
else:
    # dataset_manager.split_and_label(config.ALL_SONG_PATH,
    #                                 config.TRAINING_RATIO,
    #                                 config.TESTING_RATIO,
    #                                 config.VALIDATION_RATIO,
    #                                 config.SCALE_RATIO)

    training_paths     = utils.load('./lists/training_paths2.txt')
    training_labels    = utils.name2num(utils.load('./lists/training_labels2.txt'),tags)

    validation_paths   = utils.load('./lists/validation_paths2.txt')
    validation_labels  = utils.name2num(utils.load('./lists/validation_labels2.txt'),tags)

    x_train, num_frames_train = melspec.extract_melgrams(training_paths, config.MULTIFRAMES, trim_song=True)
    print('X_train shape:', x_train.shape)
    x_validate, num_frames_validate = melspec.extract_melgrams(validation_paths, config.MULTIFRAMES, trim_song=True)

    y_train    = np.array(training_labels)
    y_validate = np.array(validation_labels)

    utils.save_h5('./datasets/saved_melspecs/training2.h5',x_train,y_train,num_frames_train)
    utils.save_h5('./datasets/saved_melspecs/validation2.h5',x_validate,y_validate,num_frames_validate)

y_train = np_utils.to_categorical(y_train, nb_classes)
y_validate = np_utils.to_categorical(y_validate, nb_classes)

# Initialize model

if config.LOAD_WEIGHTS:
    model_path = config.WEIGHTS_PATH + "_final40_.h5"
    model = m.MusicTaggerCRNN(config.WEIGHTS_PATH, input_tensor=(1, 96, 1366), num_genres=nb_classes )
else:
    model = m.MusicTaggerCRNN("", input_tensor=(1, 96, 1366), num_genres=nb_classes )


#model = MusicTaggerCNN(weights='msd', input_tensor=(1, 96, 1366), nb)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.summary()

# # Save model architecture - if needed
#     json_string = model.to_json()
#     f = open(config.MODEL_PATH, 'w')
#     f.write(json_string)
#     f.close()


#Train model
try:
    print ("Training the model")
    f_train = open(config.MODEL_PATH+"_scores_training.txt", 'w')
    f_validate = open(config.MODEL_PATH+"_scores_validate.txt", 'w')
    f_scores = open(config.MODEL_PATH+"_scores.txt", 'w')
    time_elapsed = 0
    for epoch in range(1,config.EPOCHS+1):
        t0 = time.time()
        print ("Number of epoch: " +str(epoch)+"/"+str(config.EPOCHS))
        sys.stdout.flush()
        print ('about to crash??')
        scores = model.fit(x_train, y_train, batch_size=config.BATCH_SIZE, epochs=1, verbose=1, validation_data=(x_validate, y_validate))
        time_elapsed = time_elapsed + time.time() - t0
        print ("Time Elapsed: " +str(time_elapsed))
        sys.stdout.flush()
        score_train = model.evaluate(x_train, y_train, verbose=0)
        print('Train Loss:', score_train[0])
        print('Train Accuracy:', score_train[1])
        f_train.write(str(score_train)+"\n")
        score_validate = model.evaluate(x_validate, y_validate, verbose=0)
        print('validate Loss:', score_validate[0])
        print('validate Accuracy:', score_validate[1])
        #f_validate.write(str(score_validate)+"\n")
        #f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_validate[0])+","+str(score_validate[1]) + "\n")
        #if config.SAVE_WEIGHTS and epoch % 5 == 0:
        #    model.save_weights(config.WEIGHTS_PATH + "_epoch_" + str(epoch) + ".h5")
        #    print("Saved model to disk in: " + config.WEIGHTS_PATH + "_epoch" + str(epoch) + ".h5")
    print("before saving model")
    model.save_weights(config.WEIGHTS_PATH + "_final" + str(config.EPOCHS) + "_.h5")
    print("after saving model")
    print("Saved model to disk in: " + config.WEIGHTS_PATH + "_final" + str(config.EPOCHS) + "_.h5")

    f_train.close()
    f_validate.close()
    f_scores.close()
    # Save time elapsed
    f = open(config.MODEL_PATH+"_time_elapsed.txt", 'w')
    f.write(str(time_elapsed))
    f.close()
# Save files when an sudden close happens / ctrl C
except:
    print "exception"
    f_train.close()
    f_validate.close()
    f_scores.close()
    # Save time elapsed
    f = open(config.MODEL_PATH + "_time_elapsed.txt", 'w')
    f.write(str(time_elapsed))
    f.close()
    f_train.close()
finally:
    print "finally"
    f_validate.close()
    f_scores.close()
    # Save time elapsed
    f = open(config.MODEL_PATH + "_time_elapsed.txt", 'w')
    f.write(str(time_elapsed))
    f.close()
