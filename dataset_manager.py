# TODO : Split all songs data into training, testing and validation set
# TODO : Have a funtion to generate labels and absolute file paths for all songs in a folder. This will generate "metadata.txt" for the songs in a given folder
#        Metadata.txt will  be of format absolute_file_path|label  ( Use label "None" for no label/ songs outside the genres folder structure, and "Folder_names" as labels)
#        these Metadata.txt will go to the respective folders with names training.txt,  testing.txt , validation.txt and predict_new.txt
import os
from random import shuffle
import math

def split_and_label( allSongPath, train_ratio, test_ratio, validation_ratio):
    genres = list(os.listdir(allSongPath))
    #print genres

    weight_list = [train_ratio, test_ratio,validation_ratio]
    song_list = []

    for genre in genres:
        genre_songs = []
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
        length = len(genre_list)
        idx1 = int(math.ceil(train_ratio*length))
        idx2 = int(idx1 + math.ceil(test_ratio*length))
        training_list.extend(genre_list[0:idx1])
        testing_list.extend(genre_list[idx1+1:idx2])
        validation_list.extend(genre_list[idx2+1:]) 

    shuffle(training_list)
    shuffle(testing_list)
    shuffle(validation_list)

    training_path = open("training_paths.txt","w")
    training_label = open("training_labels.txt","w")

    testing_path = open("testing_paths.txt","w")
    testing_label = open("testing_labels.txt","w")

    validation_path = open("validation_paths.txt","w")
    validation_label = open("validation_labels.txt","w")

    for path, label in training_list:
        training_path.write(path+ "\n" )
        training_label.write(label+ "\n" )

    for path, label in testing_list:
        testing_path.write(path+ "\n" )
        testing_label.write(label+ "\n" )

    for path, label in validation_list:
        validation_path.write(path+ "\n" )
        validation_label.write(label+ "\n" )

split_and_label('./datasets/all_songs/Labelled',0.6,0.2,0.2)