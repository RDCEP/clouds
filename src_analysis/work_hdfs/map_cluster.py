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
import re

priority_txt = 'filelist_metadata_train_random-80000_nc-20_m01_b28_29_31_patches_labels_2000-2018_random_aggl.txt'
DIR_NPZ = 'output_clouds_feature_2000_2018_validfiles'

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
    npz_files = open(input_dir + '/' + txt_file, 'r').read().split('.npz')[:-1]
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


def connect_files(npy_file, npz_files, num_patches, npz_dir=DIR_NPZ):
    '''

    Inputs:
        npy_file(str): name of npy file
        npz_files: list of strings representing the filenames of the
            npz files that correspond to the npy_file
        num_patches(int): number of patches used in this clustering iteration
        npz_dir(str): directory in which the npz files are stored

    Outputs:
        info_dict:
    '''
    info_dict = {'file': [], 'indices': [], 'cluster_num': []}
    npy_array = np.load(npy_file)
    patch_counter = 0
    for npz_file in npz_files:
        match = re.search('2[0-9]*\.[0-9]*(?=\_)', npz_file)
        if match:
            filename = match.group()
            npz_filename = glob.glob(npz_dir + '/*' + str(filename) + '*.npz')[0]
            npz_array = np.load(npz_filename)
            ij_list = npz_array['clouds_xy']
            for idx in ij_list:
                    i, j = idx
                    cluster_no = npy_array[patch_counter]
                    patch_counter +=1
                    info_dict['file'].append(filename)
                    info_dict['indices'].append((i, j))
                    info_dict['cluster_num'].append(cluster_no)
                    if patch_counter == num_patches:
                        return info_dict

