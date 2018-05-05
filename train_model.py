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

import config
import utils

if config.SELECT_DEEP_MODELS:
    import model_deep as m
else:
    import model as m
# RUN ONCE ONLY AND REMEMEMBER TO SAVE MELSPECS
tags = utils.load(config.GENRES_FILE_PATH)
nb_classes = len(tags)

if config.LOAD_MELSPECS:
    x_train,  y_train, num_frames_train  = utils.load_h5(config.TRAINING_MELSPEC_FILE)
    x_validate,  y_validate, num_frames_validate  = utils.load_h5(config.VALIDATION_MELSPEC_FILE)
else:
    print "Error: No Melspec files to load"
    sys.exit()

y_train = np_utils.to_categorical(y_train, nb_classes)
y_validate = np_utils.to_categorical(y_validate, nb_classes)

# Initialize model

if config.LOAD_WEIGHTS:
    model = m.MusicTaggerCRNN(config.MODEL_WEIGHTS_FILE, input_tensor=(1, 96, 1366), num_genres=nb_classes )
else:
    model = m.MusicTaggerCRNN("", input_tensor=(1, 96, 1366), num_genres=nb_classes )

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
        f_validate.write(str(score_validate)+"\n")
        f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_validate[0])+","+str(score_validate[1]) + "\n")
    model.save_weights(config.MODEL_WEIGHTS_FILE)
    print("after saving model")
    print("Saved model to disk in: " + config.MODEL_WEIGHTS_FILE)

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
finally:
    print "finally"
    f_validate.close()
    f_scores.close()
    # Save time elapsed
    f = open(config.MODEL_PATH + "_time_elapsed.txt", 'w')
    f.write(str(time_elapsed))
    f.close()
