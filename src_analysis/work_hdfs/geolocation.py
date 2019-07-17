'''
Katy Koenig

July 2019

Functions to find latitude and longitude for patches with invalid pixels 
'''
import os
import sys
import csv
import glob
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC

hdf_libdir = '/home/koenig1/scratch-midway2/clouds/src_analysis/lib_hdfs' # change here
sys.path.insert(1, os.path.join(sys.path[0], hdf_libdir))
from alignment_lib import gen_mod35_img
import prg_StatsInvPixel as stats


def make_connecting_dict(file_csv, outputfile):
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
            bname = os.path.basename(row)
            date = bname[10:22]
            mod02 = glob.glob(mod02_dir + '/*/' + file)
            if mod02:
                mod02_path = mod02[0]
            else:
                print("No mod02 file downloaded for " + date)
            # Finds corresponding MOD)3
            mod03 = glob.glob(mod35_dir + '/*/*' + date + '*.hdf')
            if mod03:
                mod03_path = mod03[0]
                mod03_hdf = SD(mod35_path, SDC.READ)
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
                make_patches(invals_dict, mod02_path, latitude, longitude, hdf_m35)
            else:
                print("No mod35 file downloaded for " + date)
    
    results_df = pd.DataFrame(columns=['filename', 'patch_no', 'latitude', 'longitude'])
    for key in invals_dict.keys():
        patches, latitudes, longitudes, cloud_mask = invals_dict[key]
        connect_geolocation(results_df, key, patches, latitudes, longitudes, cloud_mask)
    return results_df


def make_patches(invals_dict, mod02_path, latitude, longitude):
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
    invals_dict[mod02_path] = [np.stack(patches), np.stack(latitudes),
                               np.stack(longitudes)]


def connect_geolocation(name, patches, latitudes, longitudes, clouds_mask,
                        results_df, width=128, height=128, thres=0.3):
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
                    results_df.loc[len(results_df)] = [name, patch_counter, lat, lon]
                    patch_counter += 1
            else:
                print('Null Values in ' + file)
    return results_df


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
