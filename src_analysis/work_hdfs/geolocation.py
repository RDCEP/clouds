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
from pyhdf.SD import SD, SDC

hdf_libdir = '/home/koenig1/scratch-midway2/clouds/src_analysis/lib_hdfs' # change here
sys.path.insert(1, os.path.join(sys.path[0], hdf_libdir))
from alignment_lib import gen_mod35_img
import prg_StatsInvPixel as stats


MOD02_DIRECTORY = '/home/koenig1/scratch-midway2/MOD02/clustering'
MOD03_DIRECTORY = '/home/koenig1/scratch-midway2/MOD03/clustering'
MOD35_DIRECTORY = '/home/koenig1/scratch-midway2/MOD35/clustering'
INVALIDS_CSV = 'patches_with_invalid_pixels.csv'
OUTPUT_CSV = 'output_test07172019.csv'

def make_connecting_csv(file, output=OUTPUT_CSV, mod02_dir=MOD02_DIRECTORY, 
                       mod35_dir=MOD35_DIRECTORY, mod03_dir=MOD03_DIRECTORY):
    '''

    Inputs:
        files_csv
        outputfile:

    Outputs:
    '''
    invals_dict = {}
    with open(file_csv, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            file = row[1]
            print(file)
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
                patches, latitudes, longitudes = make_patches(output, mod02_path, latitude, longitude, cloud_mask_img)
                connect_geolocation(output, mod02_path, patches, latitudes, longitudes, cloud_mask_img)
            else:
                print("No mod35 file downloaded for " + date)


def make_patches(mod02_path, latitude, longitude):
    '''

    Inputs:
        invals_dict:
        mod02_path:
        latitdue:
        longitude:

    Outputs:
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
    return np.stack(patches), np.stack(latitudes), np.stack(longitudes)


def connect_geolocation(file, output, patches, latitudes, longitudes, clouds_mask,
                        width=128, height=128, thres=0.3):
    '''

    Inputs:
        name:
        patches:
        latitudes:
        longitudes:
        clouds_mask:
        results_df:
        width:
        height:
        thres:

    Outputs:
    '''
    with open(output_file, 'a') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        nx, ny = patches.shape[:2]
        patch_counter = 0
        for i in range(nx):
            for j in range(ny):
                lat = latitudes[i, j]
                lon = longitudes[i, j]
                if not np.isnan(patches[i, j]).any():
                  if np.any(clouds_mask[i * width:(i + 1) * width,
                            j * height:(j + 1) * height] == 0):
                    tmp = clouds_mask[i * width:(i + 1) * width,
                                      j * height:(j + 1) * height]
                    nclouds = len(np.argwhere(tmp == 0))
                    if nclouds / (width * height) > thres:
                        outputwriter.writerow([file, patch_counter, lat, lon])
                        patch_counter += 1
                else:
                    print('Null Values in ' + file)
    csvfile.close()


def make_geodf(dataframe):
    '''
    Turns a dataframe with latitude and longitude columns into a geodataframe

    Inputs:
        pandas dataframe with a column that is a list of coordinates

    Outputs:
        geodataframe
    '''
    results_df['geom'] = results_df.apply(lambda row: apply_func(row['latitude'], row['longitude']), axis=1)
    results_df['geom'] = results_df['geom'].apply(geometry.Polygon)
    results_gdf = gpd.GeoDataFrame(results_df, geometry='geom')
    return results_gdf


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
    p.add_argument('--processors', type=int, default=20)
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
        last_file = completed.tail(1)['filename'].tolist()[0]
        done = completed[completed['filename'] != last_file]
        print("Last file " + last_file)
        done.to_csv(args.outputfile, index=False)
    else:
        # Initializes output csv to be appended later
        with open(args.outputfile, 'w') as csvfile:
            outputwriter = csv.writer(csvfile, delimiter=',')
            outputwriter.writerow(['filename', 'patch_no', 'latitude', 'longitude'])
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
    pool.starmap(make_connecting_csv, args_lst)
    pool.close()
    pool.join()

