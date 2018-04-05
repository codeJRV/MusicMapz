# TODO Create model(s) here 

# Updated to Keras 2 api based on https://github.com/keras-team/keras/wiki/Keras-2.0-release-notes


from __future__ import print_function
from __future__ import absolute_import

import os
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.utils.data_utils import get_file
from keras.layers import Input, Dense



def MusicTaggerCNN( weights_path='', input_tensor=None, num_genres = 10):

    channel_axis = 1
    time_axis = 3

    # Removing variations :: we are using K.image_dim_ordering() == 'th'
    # if K.image_dim_ordering() == 'tf':
    #         raise RuntimeError("Please set image_dim_ordering == 'th'."
    #                            "You can set it at ~/.keras/keras.json")

    if input_tensor is None:
        melgram_input = Input(shape=(1, 96, 1366))
    else:
        melgram_input = Input(shape=input_tensor)


    # Input block
    x = BatchNormalization(axis=time_axis, name='bn_0_freq' )(melgram_input)

    # removed trainable=False parameter as keras 2.0 api no longer supports it

    # Conv block 1
    x = Conv2D(64, (3, 3), padding="same", name="conv1")(x)
    x = BatchNormalization(axis=channel_axis,   name='bn1' )(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1' )(x)

    # Conv block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2' )(x)
    x = BatchNormalization(axis=channel_axis,   name='bn2' )(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), padding='same', name='conv3' )(x)
    x = BatchNormalization(axis=channel_axis,   name='bn3' )(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)

    # Conv block 4
    x = Conv2D(192, (3, 3), padding='same', name='conv4' )(x)
    x = BatchNormalization(axis=channel_axis,   name='bn4' )(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4' )(x)

    # Conv block 5
    x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis,   name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

    # Output
    x = Flatten(name='Flatten_1')(x)
    x = Dense(num_genres, activation='sigmoid', name='output')(x)

    model = Model(melgram_input, x)

    if  os.path.isfile(weights_path):
        model.load_weights(weights_path,by_name=True)
    else:
        print("Using default weights")

    return model

#Checking model validity
#print(MusicTaggerCNN())