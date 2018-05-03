# TODO : Split all songs data into training, testing and validation set
# TODO : Have a funtion to generate labels and absolute file paths for all songs in a folder. This will generate "metadata.txt" for the songs in a given folder
#        Metadata.txt will  be of format absolute_file_path|label  ( Use label "None" for no label/ songs outside the genres folder structure, and "Folder_names" as labels)
#        these Metadata.txt will go to the respective folders with names training.txt,  testing.txt , validation.txt and predict_new.txt
import os
from random import shuffle
import math
import json

def split_and_label( allSongPath, train_ratio, test_ratio, validation_ratio,scale_ratio = 1):
    genres = [ d for d in os.listdir(allSongPath) if os.path.isdir(os.path.join(allSongPath, d)) ]
    #print genres
    song_list = []

    genre_names = open("lists/genre_names.txt","w")

    for genre in genres:
        genre_songs = []

        genre_names.write(genre+"\n")
        song_folder = allSongPath + "/" + genre
        for path, dirs, files in os.walk(song_folder):
            for file in files:
                if file.endswith(".au"):
                    song_path = path + "/" + file
                    #print song_path
                    song = [song_path,genre]
                    genre_songs.append(song)

        song_list.append(genre_songs)

    training_list = []
    testing_list = []
    validation_list = []

    for genre_list in song_list:
        shuffle(genre_list)
        length = int(len(genre_list)*scale_ratio)
        idx1 = int(math.ceil(train_ratio*length))
        idx2 = int(idx1 + math.ceil(test_ratio*length))

        training_list.extend(genre_list[0:idx1])
        testing_list.extend(genre_list[idx1+1:idx2])
        validation_list.extend(genre_list[idx2+1:length-1])

    shuffle(training_list)
    shuffle(testing_list)
    shuffle(validation_list)

    training_path = open("lists/training_paths.txt","w")
    training_label = open("lists/training_labels.txt","w")

    testing_path = open("lists/testing_paths.txt","w")
    testing_label = open("lists/testing_labels.txt","w")

    validation_path = open("lists/validation_paths.txt","w")
    validation_label = open("lists/validation_labels.txt","w")

    all_songs_path = open("lists/all_songs_paths.txt","w")
    all_songs_label = open("lists/all_songs_labels.txt","w")

    for path, label in training_list:
        training_path.write(path+ "\n" )
        training_label.write(label+ "\n" )
        all_songs_path.write(path+ "\n" )
        all_songs_label.write(label+ "\n" )

    for path, label in testing_list:
        testing_path.write(path+ "\n" )
        testing_label.write(label+ "\n" )
        all_songs_path.write(path+ "\n" )
        all_songs_label.write(label+ "\n" )

    for path, label in validation_list:
        validation_path.write(path+ "\n" ) #genre_names = open("lists/genre_names.txt","w")
        validation_label.write(label+ "\n" )
        all_songs_path.write(path+ "\n" )
        all_songs_label.write(label+ "\n" )

    all_songs_path.close()
    all_songs_label.close()
    training_path.close()
    training_label.close()
    testing_label.close()
    testing_path.close()
    validation_label.close()
    validation_path.close()
    genre_names.close()

def split_and_label2( allSongPath, train_ratio, test_ratio, validation_ratio,scale_ratio = 1,all_song_path_file, all_song_label_file):

    genres = open(all_song_label_file,"r")
    songs  = open(all_song_path_file,"r")

    allsongdict = {}

    for i in range(len(genres))
        allsongdict[genres[i]].append(songs[i])
    
    print(allsongdict)

split_and_label2('',0,0,0,'/home/jrv/Desktop/jrv1/MusicMapz-old/lists/all_songs_paths2.txt','/home/jrv/Desktop/jrv1/MusicMapz-old/lists/all_songs_labels2.txt') 
     