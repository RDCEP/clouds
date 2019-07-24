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
import multiprocessing as mp
import numpy as np
import pandas as pd
from shapely import geometry
from pyhdf.SD import SD, SDC
from dask import dataframe as dd
from dask.multiprocessing import get

hdf_libdir = '/home/koenig1/scratch-midway2/clouds/src_analysis/lib_hdfs' # change here
sys.path.insert(1, os.path.join(sys.path[0], hdf_libdir))
from alignment_lib import gen_mod35_img
import prg_StatsInvPixel as stats


MOD02_DIRECTORY = '/home/koenig1/scratch-midway2/MOD02/clustering'
MOD03_DIRECTORY = '/home/koenig1/scratch-midway2/MOD03/clustering'
MOD35_DIRECTORY = '/home/koenig1/scratch-midway2/MOD35/clustering'
INVALIDS_CSV = 'test.csv'
OUTPUT_CSV = 'output_test.csv'
KEYS = ['filename', 'patch_no', 'latitude', 'longitude', 65535, 65534,
        65533, 65532, 65531, 65530, 65529, 65528, 65527, 65526, 65524]

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
        patches, latitudes, longitudes, fillvalue_list = make_patches(mod02_path, latitude, longitude)
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


def connect_geolocation(file, outputfile, patches, fillvalue_list, latitudes,
                        longitudes, cloud_mask, width=128, height=128,
                        thres=0.3):
    '''NEEDS EDITS
    Connects the geolocation data to each patch in an image/mod02 hdf file

    Inputs:
        file(str):
        output_file(str): csv filename to save results
        latitudes: numpy array of arrays representing latitudinal data for each
                   pixel in a patch
        longitudes: numpy array of arrays representing longitudinal data for
                    each pixel in a patch
        file(str): name of MOD02 file to be used only as identifier in CSV row
        patches: numpy array of arrays representing MOD02 patches
        clouds_mask: numpy array created from MOD35 image
        width(int): number of pixels for width of a siengle patch
        height(int): number of pixels for height of a srngle path
        thres(float): number between 0 and 1 representing the percentage
                    required of cloud cover to be considered an analyzable patch

    Outputs: Appends to existing csv file
    '''
    keys = KEYS
    codes = [65535, 65534, 65533, 65532, 65531, 65530, 65529, 65528, 65527,
             65526, 65524]
    results = {key: [] for key in keys}
    with open(outputfile, 'a') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        nx, ny = patches.shape[:2]
        patch_counter = 0
        for i in range(nx):
            for j in range(ny):
                lat = latitudes[i, j]
                lon = longitudes[i, j]
                if not np.isnan(patches[i, j]).any():
                    tmp = cloud_mask[i*width:(i+1)*width, j*height:(j+1)*height]
                    if np.any(tmp == 0):
                        nclouds = len(np.argwhere(tmp ==0))
                        if nclouds / (width * height) > thres:
                            results['filename'].append(file)
                            results['patch_no'].append(patch_counter)
                            results['latitude'].append(lat)
                            results['longitude'].append(lon)
                            for code in codes:
                                results[code].append(fillvalue_list.count(code))
                            patch_counter += 1
        results_df = pd.DataFrame.from_dict(results)
        ordered_df = find_corners(results_df[keys])
        print('Writing out for' + file)
        ordered_df.to_csv(csvfile, header=False)
    csvfile.close()


def find_corners(results_df):
    '''
    Turns a dataframe with latitude and longitude columns into a geodataframe

    Inputs:
        dataframe: pandas dataframe with a column that is a list of coordinates
        n_parts(int): number of partitions of the dataframe to be created

    Outputs: geodataframe
    '''
    # x = dd.from_pandas(dataframe, npartitions=n_parts).\
    #    map_partitions(lambda df: df.apply(lambda row: apply_func(row['latitude'], row['longitude']), axis=1), meta=pd.Series(dtype='str', name='Column X')).\
    #    compute(get=get)

    results_df['geom'] = results_df.apply(lambda row: apply_func(row['latitude'], row['longitude']), axis=1)
    results_df['geom'] = results_df['geom'].apply(geometry.Polygon)
    #results_gdf = gpd.GeoDataFrame(results_df, geometry='geom')
    return results_df.drop(columns=['latitude', 'longitude'])


def apply_func(x, y):
    '''
    Finds the four corners points of a rectangular patch

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
    return [(small_lat, small_lon), (big_lat, small_lon), (small_lat, big_lon), (big_lat, big_lon)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', type=str, default=INVALIDS_CSV)
    p.add_argument('--mod02dir', type=str, default=MOD02_DIRECTORY)
    p.add_argument('--mod35dir', type=str, default=MOD35_DIRECTORY)
    p.add_argument('--mod03dir', type=str, default=MOD03_DIRECTORY)
    p.add_argument('--processors', type=int, default=1)
    p.add_argument('--outputfile', type=str, default=OUTPUT_CSV)
    args = p.parse_args()
    print(args.processors)

    #Initializes pooling process for parallelization
    pool = mp.Pool(processes=args.processors)

    # If output csv exits, assumes cutoff by RCC so deletes last entry
    # and appends only new dates
    if os.path.exists(args.outputfile):
        print('Checking for completion')
        completed = pd.read_csv(args.outputfile)
        completed = completed[completed['filename'].notnull()]
        last_file = completed.tail(1)['filename'].tolist()
        if last_file:
            last_file = last_file[0][-44:]
        done = completed[completed['filename'] != last_file]
        done.to_csv(args.outputfile, index=False)
    else:
        # Initializes output csv to be appended later
        with open(args.outputfile, 'w') as csvfile:
            outputwriter = csv.writer(csvfile, delimiter=',')
            outputwriter.writerow(KEYS)
        csvfile.close()
        last_file = None

    args_lst = []
    filenames_df = pd.read_csv(args.input_file)
    files = list(filenames_df['filename'])
    if last_file:
        last_idx = files.index(last_file)
        files = files[last_idx:]
    for file in files:
        args_lst.append((file, args.outputfile, args.mod02dir, args.mod35dir, args.mod03dir))
    print('about to pool')
    pool.starmap(make_connecting_csv, args_lst)
    pool.close()
    pool.join()
