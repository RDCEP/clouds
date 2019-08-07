'''
Katy Koenig

August 2019

Functions to Map Clusters
(specifically cluster 0)
'''

import os
import csv
import glob
import numpy as np

priority_txt = 'filelist_metadata_train_random-80000_nc-20_m01_b28_29_31_patches_labels_2000-2018_random_aggl.txt'


def find_related_files(txt_file, input_dir):
    '''

    Inputs:
        txt_file(str):
        input_dir(str):

    Outputs:
        npy_file(str):
        npz_files(list):

    '''
    npy_file = glob.glob(input_dir + '/*' +  txt_file[37:53] + '*.npy')[0]
    npz_files = open(txt_file, 'r').read().split('.npz')[:-1]
    return npy_file, npz_files


def find_related_np(input_dir, cluster_size=80000):
    '''

    Inputs:
        input_dir(str):
        cluster_size(int):

    Outputs:

    '''
    corresponding_files = {}
    relevant_files = []
    for file in os.listdir(input_dir):
        if 'txt' and str(cluster_size) in file:
            relevant_files.append(file)
    for txt_file in relevant_files:
        npy_file, npz_files = find_related_files(txt_fle, input_dir)

        corresponding_files[txt_file] = npz_files
    return corresponding_files


def connect_files(npy_file, npz_files):
    '''

    Inputs:

    Outputs:

    '''
    npy_array = np.load(npy_file)
    for cluster_no in npy_array:
        for file in npz_files:
            npz_array = np.load(npy_file)
            ij_list = npz_array['clouds_xy']
            for indices in ij_list:
                

