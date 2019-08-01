'''
Katy Koenig

July 2019

Functions to find latitude and longitude for patches with invalid pixels
'''
import os
import sys
import csv
import glob
import argparse
import copy
import re
import multiprocessing as mp
import numpy as np
import pandas as pd
from shapely import geometry
import geopandas as gpd
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import prg_StatsInvPixel as stats

hdf_libdir = '/Users/katykoeing/Desktop/clouds/src_analysis/lib_hdfs' # change here
sys.path.insert(1, os.path.join(sys.path[0], hdf_libdir))
from alignment_lib import gen_mod35_img

MOD02_DIRECTORY = '/home/koenig1/scratch-midway2/MOD02/clustering'
MOD03_DIRECTORY = '/home/koenig1/scratch-midway2/MOD03/clustering'
MOD35_DIRECTORY = '/home/koenig1/scratch-midway2/MOD35/clustering'
INVALIDS_CSV = 'patches_with_invalid_pixels.csv'
OUTPUT_CSV = 'output_07262019.csv'
KEYS = ['filename', 'patch_no', 'latitude', 'longitude', 65535, 65534,
        65533, 65532, 65531, 65530, 65529, 65528, 65527, 65526, 65524,
        'geometry']

def make_connecting_csv(file, output=OUTPUT_CSV, mod02_dir=MOD02_DIRECTORY,
                        mod35_dir=MOD35_DIRECTORY, mod03_dir=MOD03_DIRECTORY):
    '''
    Combining functions that connects mod02, mod03 (geolocation data) and mod35
    files to create a csv with the patches that have invalid pixels and their
    corresponding latitude and longitude coordinates

    Inputs:
        file(str): name of mod02 hdf file
        output(str): name of output csv in which the individual file data
                     should be appended
        mod02_dir(str): path where MOD021KM files are located
        mod35_dir(str): path where MOD35_L2 files are located
        mod03_dir(str): path where MOD03 files are located
        Note for mod02_dir, mod35_dir and mod03_dir:
            dir = '/home/koenig1/scratch-midway2/MOD02/clustering'
            when hdf files located in
            '/home/koenig1/scratch-midway2/MOD02/clustering/clustering_laads_2000_2018_2'

    Outputs: None (appends to exisiting csv after connecting all three files)
    '''
    bname = os.path.basename(file)
    print(file)
    date = bname[10:22]
    mod02 = glob.glob(mod02_dir + '/*/' + file)
    if mod02:
        mod02_path = mod02[0]
    else:
        print("No mod02 file downloaded for " + date)
    # Finds corresponding MOD)3
    mod03 = glob.glob(mod03_dir + '/*/*' + date + '*.hdf')
    if mod03:
        mod03_path = mod03[0]
        mod03_hdf = SD(mod03_path, SDC.READ)
        lat = mod03_hdf.select('Latitude')
        latitude = lat[:, :]
        lon = mod03_hdf.select('Longitude')
        longitude = lon[:, :]
    else:
        print("No MOD03 file downloaded for " + date)
    # Finds corresponding MOD35
    mod35 = glob.glob(mod35_dir + '/*/*' + date + '*.hdf')
    if mod35:
        mod35_path = mod35[0]
        hdf_m35 = SD(mod35_path, SDC.READ)
        cloud_mask_img = stats.gen_mod35_img(hdf_m35)
        patches, latitudes, longitudes, fillvalue_list = \
                make_patches(mod02_path, latitude, longitude)
        connect_geolocation(mod02_path, output, patches, fillvalue_list,
                            latitudes, longitudes, cloud_mask_img)
    else:
        print("No mod35 file downloaded for " + date)


def make_patches(mod02_path, latitude, longitude):
    '''
    Converts data for an entire hdf image into appropriate sized patches

    Inputs:
        mod02_path(str): location of mod02 file
        latitude: numpy array of arrays with the latitudes of each pixel
        longitude: numpy array of arrays with the longitudes of each pixel

    Outputs: numpy arrays of patches, latitudes and longitudes in matching formats
    '''
    stride = 128
    patch_size = 128
    patches = []
    latitudes = []
    longitudes = []
    fillvalue_list, swath = stats.gen_mod02_img(mod02_path)
    for i in range(0, swath.shape[0], stride):
        patch_row = []
        lat_row = []
        lon_row = []
        for j in range(0, swath.shape[1], stride):
            if i + patch_size <= swath.shape[0] and j + patch_size <= swath.shape[1]:
                p = swath[i:i + patch_size, j:j + patch_size].astype(float)
                lat = latitude[i:i + patch_size, j:j + patch_size].astype(float)
                lon = longitude[i:i + patch_size, j:j + patch_size].astype(float)
                patch_row.append(p)
                lat_row.append(lat)
                lon_row.append(lon)
        if patch_row:
            patches.append(patch_row)
            latitudes.append(lat_row)
            longitudes.append(lon_row)
    return np.stack(patches), np.stack(latitudes), np.stack(longitudes), fillvalue_list


PATCH_DICT = {'MOD021KM.A2006011.1435.061.2017260224818.hdf': 80,
 'MOD021KM.A2002280.1900.061.2017182182901.hdf': 0,
 'MOD021KM.A2002135.0615.061.2017181144843.hdf': 92,
 'MOD021KM.A2006143.0230.061.2017194223145.hdf': 0,
 'MOD021KM.A2002149.0455.061.2017181155144.hdf': 18,
 'MOD021KM.A2007120.1455.061.2017248153838.hdf': 0,
 'MOD021KM.A2017110.2255.061.2017314043402.hdf': 63,
 'MOD021KM.A2017303.1525.061.2017304012556.hdf': 33,
 'MOD021KM.A2017240.0830.061.2017317014141.hdf': 80,
 'MOD021KM.A2011203.1800.061.2017325065655.hdf': 29}

def find_spec_patch(file, patches, clouds_mask, fillvalue_list, width=128, height=128, thres=0.3):
    '''
    Inputs:
        file(str): name of MOD02 file to be used only as identifier in CSV row
        patches: numpy array of arrays representing MOD02 patches
        clouds_mask: numpy array created from MOD35 image
        fillvalue_list: list of integers for each fill value
        width(int): number of pixels for width of a siengle patch
        height(int): number of pixels for height of a srngle path
        thres(float): number between 0 and 1 representing the percentage
          required of cloud cover to be considered an analyzable patch

    Outputs: None (appends to existing csv)
    '''
    nx, ny = patches.shape[:2]
        patch_counter = 0
        desired_patch = PATCH_DICT[file]
        for i in range(nx):
            for j in range(ny):
                # Indexes for matching lats/lons for each patch
                if not np.isnan(patches[i, j]).any():
                    tmp = cloud_mask[i*width:(i+1)*width, j*height:(j+1)*height]
                    if np.any(tmp == 0):
                        nclouds = len(np.argwhere(tmp == 0))
                        if nclouds / (width * height) > thres:
                            if patch_counter == desired_patch:
                                return patches[i, j]
                            patch_counter += 1


def plot_patch():
    '''
    '''
    patch = find_spec_patch(file, patches, clouds_mask, fillvalue_list)


def connect_geolocation(file, outputfile, patches, fillvalue_list, latitudes,
                        longitudes, cloud_mask, width=128, height=128,
                        thres=0.3):
    '''
    Connects the geolocation data to each patch in an image/mod02 hdf file

    Inputs:
        file(str): name of MOD02 file to be used only as identifier in CSV row
        output_file(str): csv filename to save results
        patches: numpy array of arrays representing MOD02 patches
        fillvalue_list: list of integers for each fill value
        latitudes: numpy array of arrays representing latitudinal data for each
                   pixel in a patch
        longitudes: numpy array of arrays representing longitudinal data for
                    each pixel in a patch
        clouds_mask: numpy array created from MOD35 image
        width(int): number of pixels for width of a single patch
        height(int): number of pixels for height of a single path
        thres(float): number between 0 and 1 representing the percentage
                    required of cloud cover to be considered an analyzable patch

    Outputs: Appends to existing csv file
    '''
    keys = copy.deepcopy(KEYS)
    codes = [65535, 65534, 65533, 65532, 65531, 65530, 65529, 65528, 65527,
             65526, 65524]
    keys.remove('geometry')
    # Initializes dictionary to be written to csv at end of fn
    results = {key: [] for key in keys}
    with open(outputfile, 'a') as csvfile:
        nx, ny = patches.shape[:2]
        patch_counter = 0
        for i in range(nx):
            for j in range(ny):
                # Indexes for matching lats/lons for each patch
                lat = latitudes[i, j]
                lon = longitudes[i, j]
                if not np.isnan(patches[i, j]).any():
                    tmp = cloud_mask[i*width:(i+1)*width, j*height:(j+1)*height]
                    if np.any(tmp == 0):
                        nclouds = len(np.argwhere(tmp == 0))
                        if nclouds / (width * height) > thres:
                            results['filename'].append(file[-44:])
                            results['patch_no'].append(patch_counter)
                            results['latitude'].append(lat)
                            results['longitude'].append(lon)
                            # Finds number of bands with each error code
                            for code in codes:
                                results[code].append(fillvalue_list.count(code))
                            patch_counter += 1
        results_df = pd.DataFrame.from_dict(results)
        # Gets square shape for each patch
        ordered_df = find_corners(results_df[keys])
        print('Writing out for' + file)
        ordered_df.to_csv(csvfile, header=False, index=False)
    csvfile.close()


def find_corners(results_df):
    '''
    Turns a dataframe with latitude and longitude columns into a geodataframe

    Inputs:
        dataframe: pandas dataframe with a column that is a list of coordinates
        n_parts(int): number of partitions of the dataframe to be created

    Outputs: geodataframe
    '''
    results_df['geom'] = results_df.apply(lambda row: \
                                          apply_func(row['latitude'],
                                                     row['longitude']), axis=1)
    results_df['geom'] = results_df['geom'].apply(geometry.Polygon)
    return results_df.drop(columns=['latitude', 'longitude'])


def apply_func(x, y):
    '''
    Finds the four corners points of a rectangular patch using greedy algorithm

    Inputs:
        x: the latitude column for a pandas dataframe observation
        y: the longitude column for a pandas dataframe observation
        Note that both x and y have MANY coordinates

    Outputs: a new column of a pandas dataframe with a list of four point
            coordinates
    '''
    big_lat = float("-inf")
    small_lat = float("inf")
    big_lon = float("-inf")
    small_lon = float("inf")
    for i in range(len(x[0])):
        for j in range(len(y[1])):
            curr_lat = x[i, j]
            curr_lon = y[i, j]
            if curr_lat >= big_lat:
                big_lat = curr_lat
            else:
                small_lat = curr_lat
            if curr_lon >= big_lon:
                big_lon = curr_lon
            else:
                small_lon = curr_lon
    return [(small_lat, small_lon), (big_lat, small_lon), (small_lat, big_lon),
            (big_lat, big_lon)]


def join_dataframes(coords_csv, invals_csv):
    '''
    Joins the dataframe with the number of invalid pixels per patch with the
    dataframe that contains geographic info for each patch and converts merged
    dataframe to mappable/plottable geodataframe

    Inputs:
        coords_csv: a csv file
        invals_csv: a csv file

    Outputs:
        merged_gdf: a geopandas dataframe
    '''
    # Read in CSVs and ensure no duplicate rows
    invals_df = pd.read_csv(invals_csv)
    invals_df.drop_duplicates(inplace=True)
    coords_df = pd.read_csv(coords_csv)
    coords_df.drop_duplicates(inplace=True)
    # Join two dataframes
    merged_df = pd.merge(coords_df, invals_df, on=['filename', 'patch_no'])
    # Converts geometry column to usable list of lats/lons
    # (instead of one long string)
    merged_df['geometry'] = merged_df['geometry'] \
                            .apply(lambda x: list(map(float,
                                                      re.findall('[-|0-9|\.]*[0-9]',
                                                                 x))))
    # Drops obs with invalid coordinates (e.g. latitude = -999)
    merged_df = find_invalid_lats_lons(merged_df)
    # Turns geometry column to list of tuples as (lat, lon) points
    merged_df['geometry'] = merged_df['geometry'].apply(lambda x: list(zip(x[1::2], x[::2])))
    # Turns geometry column into polygon shape to plot
    merged_df['geometry'] = merged_df['geometry'].apply(lambda x: geometry.Polygon(x))
    merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry')
    # Turns 4 corners into boundaries
    merged_gdf['geometry'] = merged_gdf['geometry'].convex_hull
    return merged_gdf


def find_invalid_lats_lons(df, col='geometry'):
    '''
    Checks latitude & longitude values of a given geometry column to ensure
    that they are valid values, i.e. latitude values are between -90 and 90
    and longitude values are between -180 and 180, and removes observations
    with invalid values

    Inputs:
        df: a pandas dataframe
        col(str): name of geometry column

    Outputs:
        new_df: a pandas dataframe
    '''
    d = {'lats': [0, 90], 'lons': [1, 180]}
    inval_idx = []
    for key in d:
        start, max_val = d[key]
        df[key] = df[col].apply(lambda x: x[start::2])
        df_name = key + '_df'
        df_name = pd.DataFrame(df[key].tolist())
        for column in df_name.columns:
            issue_idx = df_name[df_name[column] \
                        .apply(lambda x: abs(x) > max_val)].index.tolist()
            inval_idx += issue_idx
        df.drop(columns=[key], inplace=True)
    inval_set = set(inval_idx)
    new_df = df[~df.index.isin(inval_set)]
    return new_df


def create_map(dataframe, colname, img_name):
    '''
    Maps the patches with invalid pixels on a map of the world

    Inputs:
        dataframe: a geopandas dataframe
        colname(str): column name to be plotted in color gradient

    Outputs: None (saves plot to current directory as a png)
    '''
    _, ax = plt.subplots(1, figsize=(100, 100))
    df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    df.plot(ax=ax, color='white', edgecolor='black')
    dataframe.plot(ax=ax, alpha=0.1, column=colname, vmin=min(dataframe[colname]),
                   vmax=max(dataframe[colname]), cmap='summer')

    # Code for legend adjustment informed by stackoverflow response found here:
    # https://stackoverflow.com/questions/54236083/geopandas-reduce-legend-size-and-remove-white-space-below-map
    ax.set_title('Patches with Invalid Pixels', size=20)
    ax.grid()
    fig = ax.get_figure()
    cbax = fig.add_axes([0.91, 0.3, 0.03, 0.39])
    cbax.set_title('Number of Invalid Pixels', size=5)
    sm = plt.cm.ScalarMappable(cmap='summer',
                               norm=plt.Normalize(vmin=min(dataframe[colname]),
                                                  vmax=max(dataframe[colname])))
    sm._A = []
    fig.colorbar(sm, cax=cbax, format="%d")
    plt.savefig(img_name)



if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument('--input_file', type=str, default=INVALIDS_CSV)
    P.add_argument('--mod02dir', type=str, default=MOD02_DIRECTORY)
    P.add_argument('--mod35dir', type=str, default=MOD35_DIRECTORY)
    P.add_argument('--mod03dir', type=str, default=MOD03_DIRECTORY)
    P.add_argument('--processors', type=int, default=7)
    P.add_argument('--outputfile', type=str, default=OUTPUT_CSV)
    ARGS = P.parse_args()
    print(ARGS.processors)

    #Initializes pooling process for parallelization
    POOL = mp.Pool(processes=ARGS.processors)

    # If output csv exits, assumes cutoff by RCC so deletes last entry
    # and appends only new dates
    if os.path.exists(ARGS.outputfile):
        print('Checking for completion')
        COMPLETED = pd.read_csv(ARGS.outputfile)
        COMPLETED = COMPLETED[COMPLETED['filename'].notnull()]
        LAST_FILE = COMPLETED.tail(1)['filename'].tolist()
        if LAST_FILE:
            LAST_FILE = LAST_FILE[0][-44:]
        DONE = COMPLETED[COMPLETED['filename'] != LAST_FILE]
        DONE.to_csv(ARGS.outputfile, index=False)
    else:
        # Initializes output csv to be appended later
        with open(ARGS.outputfile, 'w') as csvfile:
            OUTPUTWRITER = csv.writer(csvfile, delimiter=',')
            COLS = [x for x in copy.deepcopy(KEYS) if x not in ['latitude', 'longitude']]
            OUTPUTWRITER.writerow(COLS)
        csvfile.close()
        LAST_FILE = None

    ARGS_LST = []
    FILENAMES_DF = pd.read_csv(ARGS.input_file)
    FILES = list(FILENAMES_DF['filename'])
    if LAST_FILE:
        LAST_IDX = FILES.index(LAST_FILE)
        FILES = FILES[LAST_IDX:]
    for file in FILES:
        ARGS_LST.append((file, ARGS.outputfile, ARGS.mod02dir, ARGS.mod35dir, ARGS.mod03dir))
    POOL.starmap(make_connecting_csv, ARGS_LST)
    POOL.close()
    POOL.join()
