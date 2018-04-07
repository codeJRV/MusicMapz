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
import model
import dataset_manager

import config

dataset_manager.split_and_label(config.ALL_SONG_PATH,0.6,0.2,0.2)




