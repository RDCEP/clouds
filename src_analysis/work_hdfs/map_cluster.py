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
import geopandas as gpd
from pyhdf.SD import SD, SDC
import multiprocessing as mp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#import geolocation
#import find_invalids as fi

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
             of each patch, index of patch in an image, and the cluster number
             in which the patch was assigned
    '''
    info_dict = {'file': [], 'indices': [], 'cluster_num': []}
    npy_array = np.load(npy_file)
    patch_counter = 0
    for npz_file in npz_files:
        match = re.search(r'2[0-9]*\.[0-9]*(?=\_)', npz_file)
        if match:
            #should add check to see if .npz file exits 
            npz_array = np.load(glob.glob(npz_dir + '/*' + \
                                str(match.group()) + '*.npz')[0])
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
            return None


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
            latitude, longitude = fi.gen_mod03(mod03_path, file)
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
    merged['lat'] = merged.apply(lambda x: gen_coords(x['lat'], x['indices']),
                                           axis=1)
    merged['long'] = merged.apply(lambda x: gen_coords(x['long'], x['indices']),
                                            axis=1)
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
    Combines above functions into one easily callable function, which finds all
    related files for a given txt file representing one iteration of clustering
    and saves collected info into a csv for future use
    (specifically mapping below)

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
        total_df.to_csv(output_csv, index=None)
        print('Completed csv written for ' + txt_file)
    else:
        print('Missing mod03 files for' + txt_file)
        missing_name = 'missing_mod03_' + output_csv + '.csv'
        print('Saving missing as ' + missing_name)
        missing = pd.DataFrame(missing_mod03_files, dtype='str')
        missing.to_csv(missing_name, header=None, index=False)


def find_info_all_npy(input_dir, npz_dir, mod03_dir, num_patches, nparts=4):
    '''
    Loops through directory to find relevant txt files which then finds all
    related files for a given txt file representing one iteration of clustering
    and saves collected info into a csv for future use for each txt file

    Inputs:
        input_dir(str): directory in which npy files are saved
        npz_dir(str): directory in which npz files are saved
        mod03_dir(str): directory in which MOD03 hdf files are saved
        num_patches(int): number of patches in a cluster
        nparts(int): number of partitions in which to divide a df for a given
                     npy file (must be done or RCC will disconnect)

    Outputs: None (saves csv files of info for each relevant npy file)
    '''
    for file in os.listdir(input_dir):
        if 'txt' and str(num_patches) in file:
            output_csv = file[31:-4] + '.csv'
            combine_geo(file, input_dir, npz_dir, mod03_dir, num_patches, 
                        output_csv, nparts)


#Color list from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
COLOR_LST = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4',
             '#42D4F4', '#F032E6', '#BFEF45', '#FABEBE', '#469990', '#E6BEFF',
             '#9A6324', '#FFFAC8', '#AAFFC3', '#808000', '#FFD8B1', '#000075',
             '#A9A9A9', '#800000']


def map_clusters(df, cluster_col, img_name=None):
    '''
    Maps clusters on a world mapped (for four by five maps)

    Inputs:
        df: a pandas dataframe
        cluster_col(str): column name
        img_name(str): name (and format) for saved image

    Outputs: None (saved image)
    '''
    #for cluster in sorted(df[cluster_col].unique()):
    #handle_lst = []
    _, axs = plt.subplots(nrows=4, ncols=5, figsize=(75, 75))
    counter = 0
    for row in axs:
        for col in row:
            world_df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            world_df.plot(ax=col, color='white', edgecolor='black')
            df[df[cluster_col] == counter].plot(color=COLOR_LST[counter],
                                            alpha=0.35, ax=col)
            counter += 1
            img_name = 'map_cluster0_group' + str(counter) + '.png'
            col.set_title(img_name)
    plt.tight_layout()
    plt.savefig(img_name)


def map_by_date(df, unique_col_name, cluster_col, png_name):
    '''

    Inputs:
        df: a pandas df
        unique_col_name(str):
        cluster_col(str):
        png_name(str):

    Outputs
    '''
    unique_col = df[unique_col_name].unique()
    _, axs = plt.subplots(nrows=12, ncols=6, figsize=(75, 75))
    counter = 0
    for row in axs:
        for col in row:
            if counter < (len(unique_col) - 1):
                world_df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
                world_df.plot(ax=col, color='white', edgecolor='black')
                val = unique_col[counter]
                small_df = df[df[unique_col_name] == val]
                handle_lst = []
                for cluster in sorted(small_df[cluster_col].unique()):
                    small_df[small_df[cluster_col] == cluster].plot(color=COLOR_LST[cluster], alpha=0.2, ax=col)
                    handle = mpatches.Patch(color=COLOR_LST[cluster], label=cluster)
                    handle_lst.append(handle)
                plt.legend(handles=handle_lst, bbox_to_anchor=(1.05, 1))
                counter += 1
                img_name = 'map_cluster0_group' + str(val) + '.png'
                col.set_title(img_name)
    plt.tight_layout()
    plt.savefig(png_name)


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
