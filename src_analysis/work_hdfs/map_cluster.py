'''
Katy Koenig

August 2019

Functions to Map Clusters
(specifically cluster 0)
'''
import os
import glob
import re
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC
import geolocation


PRIORITY_TXT = 'filelist_metadata_train_random-80000_nc-20_m01_b28_29_31_patches_labels_2000-2018_random_aggl.txt'
DIR_NPZ = 'output_clouds_feature_2000_2018_validfiles'

def find_related_files(txt_file, input_dir):
    '''
	Given a txt file of a list of npz files, finds the related npz files as
	well as the corresponding npy_file

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
    Locates npz files that are related to an npy file and connects the files
    to create a dictionary with the information from one clustering interation

    Inputs:
        npy_file(str): name of npy file
        npz_files: list of strings representing the filenames of the
            npz files that correspond to the npy_file
        num_patches(int): number of patches used in this clustering iteration
        npz_dir(str): directory in which the npz files are stored

    Outputs: a pandas df of the filename (as only the year/date/time info)
             of each patch, index of patch in an image, and the cluster number in which
             the patch was placed
    '''
    info_dict = {'file': [], 'indices': [], 'cluster_num': []}
    npy_array = np.load(npy_file)
    patch_counter = 0
    for npz_file in npz_files:
        match = re.search(r'2[0-9]*\.[0-9]*(?=\_)', npz_file)
        if match:
            #should add check to see if .npz file exits 
            npz_array = np.load(glob.glob(npz_dir + '/*' + str(match.group()) + '*.npz')[0])
            ij_list = npz_array['clouds_xy']
            for idx in ij_list:
                i, j = idx
                cluster_no = npy_array[patch_counter]
                patch_counter += 1
                info_dict['file'].append(match.group())
                info_dict['indices'].append((i, j))
                info_dict['cluster_num'].append(cluster_no)
                if patch_counter == num_patches:
                    return pd.DataFrame(info_dict)
        else:
            print('File name does not include correctly formatted ' + \
                  'year/date/time for npz file ' + npz_file)


def gen_mod03(mod03_path):
    '''
    Reads in MOD03 hdf file and converts data to relevant latitude and
    longitude arrays

    Inputs:
        mod03_path(str):

    Outputs:
        latitude(np array):
        longitude(np array):
    '''
    mod03_hdf = SD(mod03_path, SDC.READ)
    lat = mod03_hdf.select('Latitude')
    latitude = lat[:, :]
    lon = mod03_hdf.select('Longitude')
    longitude = lon[:, :]
    return latitude, longitude


def add_geolocation(info_df, mod03_dir):
    '''

    Inputs:
        info_df(pandas df):
        mod03_dir(str):

    Outputs:

    '''
    lst_of_files = info_df['filename'].unique()
    geo_d = {'filename': [], 'latitude': [], 'longitude': []}
    for file in lst_of_files:
        mod03_path = glob.glob(mod03_dir + '/MOD03*' + file +'*.hdf')
        latitude, longitude = gen_mod03(mod03_path)
        geo_d['filename'].append(file)
        geo_d['lat'].append(latitude)
        geo_d['long'].append(longitude)
    geo_df = pd.DataFrame(geo_d)
    merged = pd.merge(info_df, geo_df, how='left', on='filename')
    merged['lat'], merged['long'] = merged.apply(lambda x:
                                                 gen_coords(x['lat'],
                                                            x['long'],
                                                            x['indices']), axis=1)
    return geolocation.find_corners(merged, 'lat', 'lon')


def gen_coords(latitudes, longitudes, indices, patch_size=128):
    '''

    Inputs:
        latitudes():
        longitudes():
        indices(tuple):
        stride(int):
        patch_size(int):

    Outputs:
        patch_lat(np array):
        patch_lon(np array):
    '''
    i, j = indices
    start_i = i * patch_size
    end_i = (i + 1) * patch_size
    start_j = j * patch_size
    end_j = (j + 1) * patch_size
    patch_lat = latitudes[start_i:end_i, start_j:end_j].astype(float)
    patch_lon = longitudes[start_i:end_i, start_j:end_j].astype(float)
    return patch_lat, patch_lon
