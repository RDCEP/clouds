'''
Katy Koenig

August 2019

Functions to Map Clusters
'''
import os
import glob
import ast
import multiprocessing as mp
import argparse
import re
import numpy as np
import pandas as pd
#import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import geolocation as geo
import find_invalids as fi

PRIORITY_TXT = 'filelist_metadata_train_random-80000_nc-20_m01_b28_29_31_' + \
                'patches_labels_2000-2018_random_aggl.txt'
DIR_NPZ = '/home/koenig1/scratch-midway2/clusters_20/output_clouds_feature_' + \
          '2000_2018_validfiles'
INPUT_DIR = '/home/koenig1/scratch-midway2/clusters_20/group0'
MOD03_DIR = '/home/koenig1/scratch-midway2/clusters_20'


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


def connect_files(npy_file, npz_files, num_patches, npz_dir):
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
    return geo.find_corners(merged, 'lat', 'long')


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


def combine_geo(txt_file, input_dir, mod03_dir, num_patches, output_csv,
                npz_dir, nparts):
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


def find_info_all_npy(input_dir, npz_dir, mod03_dir, num_patches, nparts):
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


def map_all_continuous(dataframe, colname, img_name):
    '''
    Maps all patches in a given dataframe onto one map with a continuous
    colormap to a specified column. Image is then saved.

    Inputs:
        dataframe: a geopandas dataframe
        colname(str): column name to be plotted in color gradient
        img_name(str): name of image to be saved

    Outputs: None (saves plot to current directory as a png)
    '''
    _, ax = plt.subplots(1, figsize=(100, 100))
    df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    df.plot(ax=ax, color='white', edgecolor='black')
    dataframe.plot(ax=ax, alpha=0.1, column=colname, vmin=min(dataframe[colname]),
                   vmax=max(dataframe[colname]), cmap='summer')

    # Code for legend adjustment informed by stackoverflow response found here:
    # https://stackoverflow.com/questions/54236083/geopandas-reduce-legend-size-and-remove-white-space-below-map
    ax.set_title(f'Patches with {colname}', size=20)
    ax.grid()
    fig = ax.get_figure()
    cbax = fig.add_axes([0.91, 0.3, 0.03, 0.39])
    cbax.set_title(f'Number of {colname}', size=5)
    leg = plt.cm.ScalarMappable(cmap='summer',
                                norm=plt.Normalize(vmin=min(dataframe[colname]),
                                                   vmax=max(dataframe[colname])))
    leg._A = []
    fig.colorbar(leg, cax=cbax, format="%d")
    plt.savefig(img_name)


def map_all_discrete(df, col, png_name):
    '''
    Maps all patches in a given dataframe onto one map with different colors
    representing a unique value in a discrete column of the df and saves image.

    Note: if more than 20 distinct values (length of COLOR_LST above) in
          your chosen column, colors will be repeated

    Inputs:
        df: a geopandas dataframe
        col: column with discrete variables to differentiate by color
        png_name: desired name of saved image

    Outputs: None (saves image)
    '''
    _, axs = plt.subplots(figsize=(75, 75))
    counter = 0
    handle_lst = []
    world_df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_df.plot(ax=axs, color='white', edgecolor='black')
    for distinct in df[col].unique():
        if distinct != 'greenland':
            df[df[col] == distinct].plot(color=COLOR_LST[counter], ax=axs,
                                         alpha=0.3)
            handle = mpatches.Patch(color=COLOR_LST[counter],
                                    label=distinct)
            handle_lst.append(handle)
            counter = (counter + 1) % len(COLOR_LST)
    plt.legend(handles=handle_lst, loc='upper center',
               prop={'size': 20})
    plt.savefig(png_name)


def map_clusters(df, cluster_col, png_name):
    '''
    Maps clusters on a world mapped (for four by five maps)

    Inputs:
        df: a geoandas dataframe
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
        df: a geopandas df
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


def go(txt_file, input_dir, mod03_dir, num_patches, output_csv, npz_dir,
       nparts, mapping):
    '''
    Main driving function that connects above functions

    Inputs:
        txt_file(str): txt file with list of npz files included in clustering
            iteration
        input_dir(str): directory in which txt file is saved
        mod03_dir(str): directory in which mod03 files are saved
        num_patches(int): number of patches for clustering iteration
        output_csv(str): name of desired output csv
        npz_dir(str): directory in which npz files are saved
        nparts(int): number of partitions to use for parallelization
        mapping(list): list of desired maps

    Outputs:
        info_gdf: a geopandas dataframe with info for one clustering iteration
        (saves a csv file with info regarding each patch in a cluster)
    '''
    combine_geo(txt_file, input_dir, mod03_dir, num_patches,
                output_csv, npz_dir, nparts)
    info_df = pd.read_csv(output_csv, dtype={'file': 'str'})
    info_df['date'] = info_df['file'].apply(lambda x: x[:7])
    info_gdf = geo.clean_geom_col(info_df, 'geom')
    mapping = ast.literal_eval(mapping)
    for map_type in mapping:
        png_name = f"{output[:-4]}_{map_type}.png"
        if map_type == 'map_clusters':
            map_clusters(info_gdf, 'cluster_num', png_name)
        if map_type == 'map_by_date':
            map_by_date(info_gdf, 'date', 'cluster_num', png_name)
    return info_gdf


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument('--txt_file', type=str, default=PRIORITY_TXT)
    P.add_argument('--input_dir', type=str, default=INPUT_DIR)
    P.add_argument('--mod03_dir', type=str, default=MOD03_DIR)
    P.add_argument('--num_patches', type=int, default=80000)
    P.add_argument('--output_csv', type=str, default='output.csv')
    P.add_argument('--npz_dir', type=str, default=DIR_NPZ)
    P.add_argument('--nparts', type=int, default=7)
    P.add_argument('--map_info', type=str, default=None)
    ARGS = P.parse_args()
    go(ARGS.txt_file, ARGS.input_dir, ARGS.mod03_dir, ARGS.num_patches,
       ARGS.output_csv, ARGS.npz_dir, ARGS.nparts, ARGS.map_info)
