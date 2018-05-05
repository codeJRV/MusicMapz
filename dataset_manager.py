# TODO : Split all songs data into training, testing and validation set
# TODO : Have a funtion to generate labels and absolute file paths for all songs in a folder. This will generate "metadata.txt" for the songs in a given folder
#        Metadata.txt will  be of format absolute_file_path|label  ( Use label "None" for no label/ songs outside the genres folder structure, and "Folder_names" as labels)
#        these Metadata.txt will go to the respective folders with names training.txt,  testing.txt , validation.txt and predict_new.txt
import os
from random import shuffle
import math
import json
from random import shuffle
import math
from sklearn.model_selection import train_test_split
import json
import config
import numpy as np
import pandas as pd
import os
import ast

def get_all_song_paths_and_labels(allSongPath):
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

    all_songs_path = open(config.ALL_SONGS_PATHS,"w")
    all_songs_label = open(config.ALL_SONGS_LABELS,"w")

    for path, label in song_list:
        all_songs_path.write(path+ "\n" )
        all_songs_label.write(label+ "\n" )

    all_songs_path.close()
    all_songs_label.close()
    genre_names.close()

def get_all_song_paths_and_labels_FMA():
    load = lambda file_path: [line.rstrip('\n') for line in open(file_path)]
    name2num = lambda namelist,numlist : [numlist.index(name) for name in namelist]

    genre_map={'Classical': 'classical',
    'Hip-Hop' : 'hiphop',
    'Country' : 'country',
    'Jazz' : 'jazz',
    'Pop': 'pop',
    'Rock' : 'rock',
    'Blues' : 'blues'}

    tags=load(config.GENRES_FILE)

    genre_songs=[]
    csv_filepath=config.FMA_DATASET_CSV
    tracks = pd.read_csv(csv_filepath, index_col=0, header=[0, 1])
    print (tracks.describe())
    small = tracks['set', 'subset'] <= 'small'
    tracks_dict = tracks.loc[small, ('track', 'genre_top')]
    print (tracks_dict)

    #print (tracks_dict.get_value(2))
    #print (tracks.describe())
    #print (tracks.columns)
    #print tracks[2]['track']['track_id']
    #print set(tracks['track']['genre_top'])

    all_songs_path = open(config.ALL_SONGS_PATHS,"w")
    all_songs_label = open(config.ALL_SONGS_LABELS,"w")

    song_folder = config.SONG_FLODER_FMA # EDIT

    for path, dirs, files in os.walk(song_folder):
        for file in files:
            if file.endswith(".mp3"):
                song_path = path + "/" + file
                #print song_path
                song_id=int(song_path[-10:-4])
                print (song_id)
                genre= tracks_dict.get_value(song_id)
                if genre in genre_map:
                    song = [song_path,genre_map[genre]]
                    genre_songs.append(song)

    for path, label in genre_songs:
        all_songs_path.write(path+ "\n")
        all_songs_label.write(label+ "\n")
