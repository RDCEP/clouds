'''
Katy Koenig

August 2019

Functions to Map Clusters
'''
import os
import glob
import ast
import multiprocessing as mp
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import geolocation as geo
import find_invalids as fi

PRIORITY_TXT = 'filelist_metadata_train_random-80000_nc-20_m01_b28_29_31_' + \
                'patches_labels_2000-2018_random_aggl.txt'
DIR_NPZ = '/home/koenig1/scratch-midway2/clusters_20/output_clouds_feature_' + \
          '2000_2018_validfiles'
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
    npy_file = glob.glob(f'{input_dir}/*{txt_file[37:53]}*.npy')[0]
    npz_files = open(f'{input_dir }/{txt_file}', 'r').read().split('.npz')[:-1]
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
            npz_array = np.load(glob.glob(f'{npz_dir}/*{str(match.group())}*.npz')[0])
            ij_list = npz_array['clouds_xy']
            for idx in ij_list:
                while patch_counter < num_patches:
                    i, j = idx
                    cluster_no = npy_array[patch_counter]
                    patch_counter += 1
                    info_dict['file'].append(match.group())
                    info_dict['indices'].append((i, j))
                    info_dict['cluster_num'].append(cluster_no)
                #if patch_counter == num_patches:
                return pd.DataFrame(info_dict)
        else:
            print('File name does not include correctly formatted ' + \
                  f'year/date/time for npz file {npz_file}')
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
        found_file = glob.glob(f'{mod03_dir}/MOD03*{file}*.hdf')
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
    return geo.find_corners(merged)


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
    if len(indices) == 2:
        i, j = indices
    else:
        i, j = ast.literal_eval(indices)
    start_i = i * patch_size
    end_i = (i + 1) * patch_size
    start_j = j * patch_size
    end_j = (j + 1) * patch_size
    patch_geo_info = geo_col[start_i:end_i, start_j:end_j].astype(float)
    return patch_geo_info


def combine_geo(txt_file, input_dir, mod03_dir, num_patches,
                output_csv, npz_dir=DIR_NPZ, nparts=7):
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
    info_df = connect_files(npy_file, npz_files, num_patches, npz_dir)
    geo_df, missing_mod03_files = get_geo_df(info_df, mod03_dir)
    # Check if any MOD03 files are needed to have completed clustering iteration
    if not missing_mod03_files:
        merged = pd.merge(info_df, geo_df, how='left', on='file')
        # Have to break up dataframe or else will get kicked off RCC
        num_rows = merged.shape[0] / 4
        for i in range(4):
            df_name = f'df_{str(i)}'
            df_name = merged.iloc[int(i * num_rows):int((i + 1) * num_rows)]
            # Parallelizing each df
            data_split = np.array_split(df_name, nparts)
            pool = mp.Pool(nparts)
            processed_df = pd.concat(pool.map(get_specific_geo, data_split))
            pool.close()
            pool.join()
            all_dfs.append(processed_df)
        total_df = pd.concat(all_dfs)
        total_df.to_csv(output_csv, index=None)
    else:
        print(f'Missing mod03 files for {txt_file}')
        missing_name = f'missing_mod03_{output_csv}.csv'
        print(f'Saving missing as {missing_name}')
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
            output_csv = f'{file[31:-4]}.csv'
            combine_geo(file, input_dir, mod03_dir, num_patches,
                        output_csv, npz_dir, nparts)


#Color list from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
COLOR_LST = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4',
             '#42D4F4', '#F032E6', '#BFEF45', '#FABEBE', '#469990', '#E6BEFF',
             '#9A6324', '#FFFAC8', '#AAFFC3', '#808000', '#FFD8B1', '#000075',
             '#A9A9A9', '#800000']


def map_clusters(df, cluster_col, png_name):
    '''
    Maps clusters on a world mapped (for four by five maps)

    Inputs:
        df: a pandas dataframe
        cluster_col(str): column name
        png_name(str): name (and format) for saved image

    Outputs: None (saved image)
    '''
    _, axs = plt.subplots(nrows=4, ncols=5, figsize=(75, 75))
    counter = 0
    # Iterates through axs to get x by y number of plots in one image 
    # (instead of 1 long row or col)
    for row in axs:
        for col in row:
            # Gets simple world map as base for plotting
            world_df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            world_df.plot(ax=col, color='white', edgecolor='black')
            # Filters df to correct cluster num. for plotting
            # Plots w/ corresponding color slice from list above fn
            df[df[cluster_col] == counter].plot(color=COLOR_LST[counter],
                                                alpha=0.5, ax=col)
            # Increases counter to up cluster num.
            counter += 1
            img_name = f'map_cluster0_group {str(counter)}'
            col.set_title(img_name)
    plt.tight_layout()
    plt.savefig(png_name)


def map_by_date(df, unique_col_name, cluster_col, png_name):
    '''
    Maps subplots of dataframe, with each subplot being a unique value in
    the unique_col_name column (usually date). In each plot, the colors relate
    to the cluster_col (cluster number). The colors are each cluster number are
    invariant across each plot (e.g. if cluster 1 is red in the first plot, it
    is also red in the second plot).

    Inputs:
        df: a pandas df
        unique_col_name(str): column for each subplot (usually date)
        cluster_col(str): column with cluster numbers
        png_name(str): name of image to be saved

    Outputs None (saves a png image)
    '''
    unique_col = df[unique_col_name].unique()
    _, axs = plt.subplots(nrows=12, ncols=6, figsize=(75, 75))
    counter = 0
    # Iterates through axs to get x by y number of plots in one image 
    # (instead of 1 long row or col)
    for row in axs:
        for col in row:
            if counter < (len(unique_col) - 1):
                # Gets simple world map as base for plotting
                world_df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
                world_df.plot(ax=col, color='white', edgecolor='black')
                val = unique_col[counter]
                # Filters for appropriate date
                small_df = df[df[unique_col_name] == val]
                handle_lst = []
                # Essentially this is a bunch of plots
                # (one for each cluster num.) on top of one another 
                for cluster in sorted(small_df[cluster_col].unique()):
                    small_df[small_df[cluster_col] == cluster]. \
                             plot(color=COLOR_LST[cluster], alpha=0.6, ax=col)
                    handle = mpatches.Patch(color=COLOR_LST[cluster],
                                            label=cluster)
                    handle_lst.append(handle)
                counter += 1
                # Sets name for individual image
                img_name = f'map_cluster0_group{str(val)}.png'
                col.set_title(img_name)
    # Sets one legend for all the plots
    plt.legend(handles=handle_lst, loc='upper center', ncol=2,
               prop={'size': 20})
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
    df = pd.read_csv(csv_name, dtype={'file': 'str'})
    gdf = geo.clean_geom_col(df, 'geom')
    map_clusters(gdf, cluster_col, img_name)
    return gdf
