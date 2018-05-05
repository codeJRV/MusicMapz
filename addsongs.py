import numpy as np
import pandas as pd
import os
import ast
load = lambda file_path: [line.rstrip('\n') for line in open(file_path)]
name2num = lambda namelist,numlist : [numlist.index(name) for name in namelist]

genre_map={'Classical': 'classical',
'Hip-Hop' : 'hiphop',
'Country' : 'country',
'Jazz' : 'jazz',
'Pop': 'pop',
'Rock' : 'rock',
'Blues' : 'blues'}

tags=load('/home/jrv/Desktop/jrv1/MusicMapz-old/lists/genre_names.txt')

genre_songs=[]
csv_filepath='/home/jrv/Desktop/jrv1/fma/tracks.csv'
tracks = pd.read_csv(csv_filepath, index_col=0, header=[0, 1])
print (tracks.describe())
small = tracks['set', 'subset'] <= 'small'
tracks_dict = tracks.loc[small, ('track', 'genre_top')]
print (tracks_dict)

all_songs_path = open("/home/jrv/Desktop/jrv1/MusicMapz-old/lists/small_genre.txt","w")
for song in tracks_dict:
    all_songs_path.write(str(song)+ "\n")

#print (tracks_dict.get_value(2))
#print (tracks.describe())
#print (tracks.columns)
#print tracks[2]['track']['track_id']
#print set(tracks['track']['genre_top'])

all_songs_path = open("/home/jrv/Desktop/jrv1/MusicMapz-old/lists/all_songs_paths2.txt","w")
all_songs_label = open("/home/jrv/Desktop/jrv1/MusicMapz-old/lists/all_songs_labels2.txt","w")

song_folder = '/media/jrv/Data/fma_large'# EDIT

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
    all_songs_path.write(path+ "\n" )
    all_songs_label.write(label+ "\n" )
