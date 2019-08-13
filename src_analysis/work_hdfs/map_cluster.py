'''
Katy Koenig

August 2019

Functions to Map Clusters
(specifically cluster 0)
'''
import os
import glob
import re
import ast
import numpy as np
import pandas as pd
#import geopandas as gpd
from pyhdf.SD import SD, SDC
import multiprocessing as mp
import geolocation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PRIORITY_TXT = 'filelist_metadata_train_random-80000_nc-20_m01_b28_29_31_patches_labels_2000-2018_random_aggl.txt'
DIR_NPZ = '/home/koenig1/scratch-midway2/clusters_20/output_clouds_feature_2000_2018_validfiles'
INPUT_DIR = '/home/koenig1/scratch-midway2/clusters_20/group0'


def find_related_files(txt_file, input_dir):
    '''
    Given a txt file of a list of npz files, finds the related npz files as
    well as the corresponding npy_file

    Inputs:
        txt_file(str): txt file representing one iteration of clustering
        input_dir(str): directory in which npy files are saved

    Outputs:
        npy_file(str): name of numpy file for cluster iteration
        npz_files(list): list of npz files related to the npy file

    '''
    npy_file = glob.glob(input_dir + '/*' +  txt_file[37:53] + '*.npy')[0]
    npz_files = open(input_dir + '/' + txt_file, 'r').read().split('.npz')[:-1]
    return npy_file, npz_files


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
        mod03_path(str): path and filename of mod03 hdf file

    Outputs:
        latitude: numpy array of arrays with the latitudes of each pixel
        longitude: numpy array of arrays with the longitudes of each pixel
    '''
    mod03_hdf = SD(mod03_path, SDC.READ)
    lat = mod03_hdf.select('Latitude')
    latitude = lat[:, :]
    lon = mod03_hdf.select('Longitude')
    longitude = lon[:, :]
    return latitude, longitude


def get_geo_df(info_df, mod03_dir):
    '''
    Creates a dataframe of the geographic information (from MOD03 hdf files)
    to match the files in info_df

    Inputs:
        info_df: pandas dataframe with each row representing a path with its
                 data/time location (file col), indices and cluster number
        mod03_dir(str): folder in which mod03 files are saved

    Outputs:
        geo_df: pandas dataframe with the lat/lon information for each file
                in info_df
        missing_mod03_files: list of files from the file col in info_df for
                             which the MOD03 file has not yet been downloaded
    '''
    missing_mod03_files = []
    lst_of_files = info_df['file'].unique()
    geo_d = {'file': [], 'lat': [], 'long': []}
    for file in lst_of_files:
        found_file = glob.glob(mod03_dir + '/MOD03*' + file +'*.hdf')
        if found_file:
            mod03_path = found_file[0]
            latitude, longitude = gen_mod03(mod03_path)
            geo_d['file'].append(file)
            geo_d['lat'].append(latitude)
            geo_d['long'].append(longitude)
        else:
            missing_mod03_files.append(file)
    geo_df = pd.DataFrame(geo_d)
    return geo_df, missing_mod03_files


def get_specific_geo(merged):
    '''
    Locates the patch-specific latitudinal and longitudinal data and finds
    the four corners of each patch

    Inputs:
        merged: a pandas dataframe

    Outputs:
        a pandas dataframe with a 'geom' column as the column for representing
        patch location
    '''
    merged['lat'] = merged.apply(lambda x: gen_coords(x['lat'], x['indices']), axis=1)
    merged['long'] = merged.apply(lambda x: gen_coords(x['long'], x['indices']), axis=1)
    return geolocation.find_corners(merged, 'lat', 'long')


def gen_coords(geo_col, indices, patch_size=128):
    '''
    Locates the patch-specific location details
    (either lat or long depending on input column)

    Inputs:
        geo_col(str): column with array of arrays
        indices(tuple): (x, y) of related patch
        patch_size(int): number of pixels in a patch

    Outputs:
        patch_geo_info: np array with patch-specific info
    '''
    i, j = indices
    start_i = i * patch_size
    end_i = (i + 1) * patch_size
    start_j = j * patch_size
    end_j = (j + 1) * patch_size
    patch_geo_info = geo_col[start_i:end_i, start_j:end_j].astype(float)
    return patch_geo_info


def combine_geo(txt_file, input_dir, npz_dir, mod03_dir,
                              num_patches, output_csv, nparts=4):
    '''
    Combines above functions into one easily callable function, which files all
    related files for a given txt file representing one iteration of clustering
    and saves collected info into a csv for future use (specifically mapping below)

    Inputs:
        txt_file(str): txt file representing one iteration of clustering
        input_dir(str): directory in which npy files are saved
        npz_dir(str): directory in which npz files are saved
        mod03_dir(str): directory in which MOD03 hdf files are saved
        num_patches(int): number of patches in a cluster
        output_csv(str): name for output csv

    Outputs: A saved csv
    '''
    all_dfs = []
    npy_file, npz_files = find_related_files(txt_file, input_dir)
    info_df = connect_files(npy_file, npz_files, num_patches, npz_dir=DIR_NPZ)
    geo_df, missing_mod03_files = get_geo_df(info_df, mod03_dir)
    if not missing_mod03_files:
        merged = pd.merge(info_df, geo_df, how='left', on='file')
        num_rows = merged.shape[0] / nparts
        for i in range(nparts):
            df_name = 'df_' + str(i)
            df_name  = merged.iloc[int(i * num_rows):int((i + 1) * num_rows)]
            df_name = get_specific_geo(df_name)
            all_dfs.append(df_name)
        total_df = pd.concat(all_dfs)

    #     total_df.to_csv(output_csv, index=None)
    # else:
    #     print('Missing mod03 files')
    #     print('Saving missing as csv named missing_mod03.csv')
    #     missing = pd.DataFrame(missing_mod03_files, dtype='str')
    #     missing.to_csv('missing_mod03.csv', header=None, index=False)
    return total_df


def find_related_np(input_dir, cluster_size=80000):
    '''
    TO BE EDITED: for looping over all given txt files

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



# Color list from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
COLOR_LST = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4',
             '#42D4F4', '#F032E6', '#BFEF45', '#FABEBE', '#469990', '#E6BEFF',
             '#9A6324', '#FFFAC8', '#AAFFC3', '#808000', '#FFD8B1', '#000075',
             '#A9A9A9', '#800000']


def map_clusters(df, cluster_col, img_name):
    '''
    Maps clusters on a world mapped

    Inputs:
        df: a pandas dataframe
        cluster_col(str): column name
        img_name(str): name (and format) for saved image

    Outputs: None (saved image)
    '''
    _, ax = plt.subplots(figsize=(100, 100))
    world_df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_df.plot(ax=ax, color='white', edgecolor='black')
    handle_lst = []
    for cluster in sorted(df[cluster_col].unique()):
        df[df[cluster_col] == cluster].plot(color=COLOR_LST[cluster], alpha=0.35, ax=ax)
        handle = mpatches.Patch(color=COLOR_LST[cluster], label=cluster)
        handle_lst.append(handle)
    plt.legend(handles=handle_lst, bbox_to_anchor=(1.05, 1))
    plt.savefig(img_name)


def clean_and_plot(csv_name, img_name, cluster_col='cluster_no'):
    '''
    Reads in csv, cleans geometry column to be usable for plotting and maps
    clusters on a global map.

    Inputs:
        csv_name(str): name of csv with information to be plotted
        img_name(str): name of image to be saved
        cluster_col(str): col in csv associated with cluster number

    Outputs: a geopandas dataframe and a saved map
    '''
    df = pd.read_csv(csv_name)
    gdf = geo.clean_geom_col(df, 'geom')
    map_clusters(gdf, cluster_col, img_name)
    return gdf
