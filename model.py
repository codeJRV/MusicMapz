# TODO Create model(s) here 

from __future__ import print_function
from __future__ import absolute_import

import os
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.utils.data_utils import get_file
from keras.layers import Input, Dense



def MusicTaggerCNN(weights_path, load_weights=False, input_tensor=None, num_genres = 10):

    channel_axis = 1
    time_axis = 3

    # Removing variations :: we are using K.image_dim_ordering() == 'th'
    if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")

    if input_tensor is None:
        melgram_input = Input(shape=(1, 96, 1366))
    else:
        melgram_input = Input(shape=input_tensor)


    # Input block
    x = BatchNormalization(axis=time_axis, name='bn_0_freq', trainable=False)(melgram_input)

    # Conv block 1
    x = Convolution2D(64, (3, 3), padding="same", trainable=False, name="conv1")(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1', trainable=False)(x)

    # Conv block 2
    x = Convolution2D(128, (3, 3), border_mode='same', name='conv2', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, (3, 3), border_mode='same', name='conv3', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(192, (3, 3), border_mode='same', name='conv4', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4', trainable=False)(x)

    # Conv block 5
    x = Convolution2D(256, (3, 3), border_mode='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

    # Output
    x = Flatten(name='Flatten_1')(x)
    x = Dense(num_genres, activation='sigmoid', name='output')(x)

    model = Model(melgram_input, x)

    if load_weights and os.path.isfile(weights_path):
        model.load_weights(weights_path,by_name=True)
    else:
        raise ValueError('Invalid Path')

    return model
