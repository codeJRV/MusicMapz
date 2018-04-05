# TODO : Given a bunch of songs path, send it thru the model and generate predicted feature vectors here. Use this for doing TSNE

## Dummy code to create features.txt

import os
import pickle
import tensorflow as tf
from tensorboard.plugins import projector

import numpy as np

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
