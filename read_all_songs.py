from random import shuffle
import math
from sklearn.model_selection import train_test_split
import json

def split_and_label2( allSongPath, train_ratio, test_ratio, validation_ratio,scale_ratio,all_song_path_file, all_song_label_file):


    with open(all_song_label_file) as f:
        genres = f.read().splitlines()

    with open(all_song_path_file) as g:
        songs = g.read().splitlines()


    X_train, X_test, y_train, y_test = train_test_split(songs, genres, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


    training_path = "lists/training_paths2.txt"
    training_label = "lists/training_labels2.txt"

    testing_path = "lists/testing_paths2.txt"
    testing_label = "lists/testing_labels2.txt"

    validation_path = "lists/validation_paths2.txt"
    validation_label = "lists/validation_labels2.txt"

    # all_songs_path = open("lists/all_songs_paths.txt","w")
    # all_songs_label = open("lists/all_songs_labels.txt","w")


    with open (training_path,"w")as fp:
        for line in X_train:
            fp.write(line+"\n")
    with open (training_label,"w")as fp:
        for line in y_train:
            fp.write(line+"\n")    

    with open (testing_path,"w")as fp:
        for line in X_test:
            fp.write(line+"\n")
    with open (testing_label,"w")as fp:
        for line in y_test:
            fp.write(line+"\n")


    with open (validation_path,"w")as fp:
        for line in X_val:
            fp.write(line+"\n")
    with open (validation_label,"w")as fp:
        for line in y_val:
            fp.write(line+"\n")

    print (len(genres))
    print ( len (X_train))
    print ( len (X_test))
    print (len(X_val))

    allsongdict = {}

    for genre in genres:
        if not (genre in allsongdict):
            allsongdict[genre] = []

    for i in range(len(genres)):
        allsongdict[genres[i]].append(songs[i])
    
    for key in allsongdict:
        print ( key, ':', len(allsongdict[key]))
        shuffle(allsongdict[key])





split_and_label2('',0,0,0,0,'/home/jrv/Desktop/jrv1/MusicMapz-old/lists/all_songs_paths2.txt','/home/jrv/Desktop/jrv1/MusicMapz-old/lists/all_songs_labels2.txt') 
