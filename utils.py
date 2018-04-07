# TODO : all Load functions

import os
import numpy as np

import h5py

load = lambda file_path: [line.rstrip('\n') for line in open(file_path)] 
name2num = lambda namelist,numlist : [numlist.index(name) for name in namelist]

def load_h5(file_path):
    with h5py.File(file_path, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('labels'))
        num_frames = np.array(hf.get('num_frames'))
    return data, labels, num_frames


def save_h5(path, data, labels, num_frames):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('labels', data=labels)
        hf.create_dataset('num_frames', data=num_frames)
