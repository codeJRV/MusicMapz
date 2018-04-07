# TODO : Training Function ( use the training set and validation set, do not touch the testing set)

from random import shuffle

song_list = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]

for genre_list in song_list:
    print genre_list
    genre_list = shuffle(genre_list)
    
print song_list[2]