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
from functools import partial
import numpy as np
import pandas as pd
#import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj
import shapely.ops as ops
from shapely import geometry
import find_invalids as fi
import prg_StatsInvPixel as stats


HDF_LIBDIR = '/Users/katykoeing/Desktop/clouds/src_analysis/lib_hdfs' #change here
sys.path.insert(1, os.path.join(sys.path[0], HDF_LIBDIR))
#from alignment_lib import gen_mod35_img

# Put in your corresponding directories below
MOD02_DIRECTORY = '/home/koenig1/scratch-midway2/big_invalids/mod02invalids'
MOD03_DIRECTORY = '/home/koenig1/scratch-midway2/big_invalids/mod03invalids'
MOD35_DIRECTORY = '/home/koenig1/scratch-midway2/big_invalids/mod35invalids'
INVALIDS_CSV = '/home/koenig1/clouds/src_analysis/combined/mod02_files_big_patches.csv'
OUTPUT_CSV = 'corrected_invalids.csv'
KEYS = ['filename', 'patch_no', 'invalid_pixels', 'latitude', 'longitude',
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
    # Finds corresponding MOD03
    mod03_glob = glob.glob(f'{mod03_dir}/*{date}*.hdf')
    latitude, longitude = fi.gen_mod03(mod03_glob, date)
    # Makes mod02 patches
    mod02_glob = glob.glob(f'{mod02_dir}/*{file}')
    patches, fillvalue_list, latitudes, longitudes = fi.gen_mod02(mod02_glob,
                                                                  file,
                                                                  latitude,
                                                                  longitude)
    # Finds corresponding MOD35
    mod35_glob = glob.glob(f'{mod35_dir}/*{date}*.hdf')
    print(mod35_glob)
    cloud_mask_img = fi.gen_mod35(mod35_glob, date)
    connect_geolocation(mod02_glob[0], output, patches, fillvalue_list,
                        latitudes, longitudes, cloud_mask_img)


def make_patches(mod02_path, latitude=None, longitude=None):
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
                patch = swath[i:i + patch_size, j:j + patch_size].astype(float)
                if isinstance(latitude, np.ndarray):
                    lat = latitude[i:i + patch_size,
                                   j:j + patch_size].astype(float)
                    lon = longitude[i:i + patch_size,
                                    j:j + patch_size].astype(float)
                    lat_row.append(lat)
                    lon_row.append(lon)
                patch_row.append(patch)
        if patch_row:
            patches.append(patch_row)
            latitudes.append(lat_row)
            longitudes.append(lon_row)
    return np.stack(patches), fillvalue_list, np.stack(latitudes), \
           np.stack(longitudes)


PATCH_DICT = {'MOD021KM.A2006011.1435.061.2017260224818.hdf':
              ['MOD35_L2.A2006011.1435.061.2017261011021.hdf', 80],
              'MOD021KM.A2002280.1900.061.2017182182901.hdf':
              ['MOD35_L2.A2002280.1900.061.2017253233720.hdf', 0],
              'MOD021KM.A2002135.0615.061.2017181144843.hdf':
              ['MOD35_L2.A2002135.0615.061.2017251030517.hdf', 92],
              'MOD021KM.A2006143.0230.061.2017194223145.hdf':
              ['MOD35_L2.A2006143.0230.061.2017264231242.hdf', 0],
              'MOD021KM.A2002149.0455.061.2017181155144.hdf':
              ['MOD35_L2.A2002149.0455.061.2017251161148.hdf', 18],
              'MOD021KM.A2007120.1455.061.2017248153838.hdf':
              ['MOD35_L2.A2007120.1455.061.2017280090632.hdf', 0],
              'MOD021KM.A2017110.2255.061.2017314043402.hdf':
              ['MOD35_L2.A2017110.2255.061.2017314070909.hdf', 63],
              'MOD021KM.A2017303.1525.061.2017304012556.hdf':
              ['MOD35_L2.A2017303.1525.061.2017304012716.hdf', 33],
              'MOD021KM.A2017240.0830.061.2017317014141.hdf':
              ['MOD35_L2.A2017240.0830.061.2017317025921.hdf', 80],
              'MOD021KM.A2011203.1800.061.2017325065655.hdf':
              ['MOD35_L2.A2011203.1800.061.2017325083429.hdf', 29]}


def find_spec_patch(file, patches, cloud_mask, width=128,
                    height=128, thres=0.3):
    '''
    Finds as specific patch in a cloud image as specified by the patch
    count in the MOD02 filename

    Inputs:
        file(str): name of MOD02 file to be used only as identifier in CSV row
        patches: numpy array of arrays representing MOD02 patches
        clouds_mask: numpy array created from MOD35 image
        width(int): number of pixels for width of a single patch
        height(int): number of pixels for height of a single path
        thres(float): number between 0 and 1 representing the percentage
                      required of cloud cover to be considered an analyzable
                      patch

    Outputs:
        Specific patch (array of arrays) in an image as described in the inputs
    '''
    x_pixels, y_pixels = patches.shape[:2]
    patch_counter = 0
    value = PATCH_DICT[file]
    desired_patch = value[1]
    for i in range(x_pixels):
        for j in range(y_pixels):
            # Indexes for matching lats/lons for each patch
            if not np.isnan(patches[i, j]).any():
                tmp = cloud_mask[i * width:(i + 1) * width,
                                 j * height:(j + 1) * height]
                if np.any(tmp == 0):
                    nclouds = len(np.argwhere(tmp == 0))
                    if nclouds / (width * height) > thres:
                        if patch_counter == desired_patch:
                            return patches[i, j]
                        patch_counter += 1
            return None


def plot_patches(file_dir, patch_d=PATCH_DICT):
    '''
    Saves plot of specific patch

    Inputs:
        file_dir(str): name of directory in which MOD35 files are stored
        patch_d(dict): dictionary which connects MOD02 filename to MOD35
                       filename and the patch number to be plotted

    Outputs: Saved png files of cloud images
    '''
    for key, val in patch_d.items():
        mod02_path = file_dir + key
        mod35_path = file_dir + val[0]
        patches, _, _, _ = make_patches(mod02_path)
        cloud_mask = fi.gen_mod35(mod35_path)
        patch = find_spec_patch(key, patches, cloud_mask)
        _, axs = plt.subplots(nrows=6, ncols=1, figsize=(15, 15))
        for ax, interp in zip(axs, range(6)):
            ax.imshow(patch[:, :, interp], cmap='inferno')
            ax.set_title(key[10:17] + '_' + str(interp), fontsize=8)
        plt.savefig(f'clouds-imgs/{key[10:17]}.png')
        print(f"Completed: {key[10:17]}")


def connect_geolocation(file, outputfile, patches, fillvalue_list, latitudes,
                        longitudes, cloud_mask, nparts=1, width=128, height=128,
                        thres=0.3, sdsmax=32767):
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
    keys.remove('geometry')
    # Initializes dictionary to be written to csv at end of fn
    results = {key: [] for key in keys}
    with open(outputfile, 'a') as csvfile:
        x_pixels, y_pixels = patches.shape[:2]
        patch_counter = 0
        for i in range(x_pixels):
            for j in range(y_pixels):
                # Indexes for matching lats/lons for each patch
                lat = latitudes[i, j]
                lon = longitudes[i, j]
                if not np.isnan(patches[i, j]).any():
                    tmp = cloud_mask[i * width:(i + 1) * width,
                                     j * height:(j + 1) * height]
                    if np.any(tmp == 0):
                        nclouds = len(np.argwhere(tmp == 0))
                        if nclouds / (width * height) > thres:
                            n_inv_pixel = 0
                            for iband in range(6):
                                tmp_array = patches[i, j, :, :, iband]
                                tmp_fillvalue = fillvalue_list[iband]
                                err_idx = np.where((tmp_array >= sdsmax) & \
                                          (tmp_array < tmp_fillvalue))
                                n_inv_pixel += len(err_idx[0])
                            results['filename'].append(file)
                            results['patch_no'].append(patch_counter)
                            results['invalid_pixels'].append(n_inv_pixel)
                            results['latitude'].append(lat)
                            results['longitude'].append(lon)
                            patch_counter += 1
        results_df = pd.DataFrame.from_dict(results)
        ordered_df = find_corners(results_df[keys])
        # Parallelizes finding 4 corners
        #data_split = np.array_split(results_df[keys], nparts)
        #pool = mp.Pool(nparts)
        #ordered_df = pd.concat(pool.map(find_corners, data_split))
        #pool.close()
        #pool.join()
        print(f'Writing out for {file}')
        ordered_df.to_csv(csvfile, header=False, index=False)
    csvfile.close()


def find_corners(results_df, lat_col='latitude', lon_col='longitude'):
    '''
    Converts latitude and longitude columns into more usable geom column with
    lat/long data of the four corners of each patch

    Inputs:
        dataframe: pandas dataframe with a column that is a list of coordinates

    Outputs: a pandas dataframe with a 'geom' column
    '''
    results_df['geom'] = results_df.apply(lambda row: \
                                          apply_func_corners(row[lat_col],
                                                             row[lon_col]),
                                          axis=1)
    results_df.drop(columns=[lat_col, lon_col])
    two_vals = results_df[results_df['geom'].apply(lambda x: len(x) == 2)]. \
                                             index.to_list()
    normal_df = results_df[~results_df.index.isin(two_vals)]
    issue_df = results_df[results_df.index.isin(two_vals)]
    new_df = pd.DataFrame(columns=issue_df.columns)
    # Creates two rows if patch needs to be broken up due to change from - to + 
    # longitude (or vice versa)
    for idx in two_vals:
        reg_cols = issue_df[issue_df.columns.difference(['geom'])].loc[idx]
        new_row = reg_cols.to_dict()
        new_row['geom'] = issue_df['geom'].loc[idx][0]
        new_df.loc[len(new_df)] = new_row
        new_row['geom'] = issue_df['geom'].loc[idx][1]
        new_df.loc[len(new_df)] = new_row
    total_df = pd.concat([normal_df, new_df])
    return total_df.drop(columns=[lat_col, lon_col])


def apply_func_corners(lats, lons):
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
    for i in range(len(lats[0])):
        for j in range(len(lons[1])):
            curr_lat = lats[i, j]
            curr_lon = lons[i, j]
            if curr_lat >= big_lat:
                big_lat = curr_lat
            else:
                small_lat = curr_lat
            if curr_lon >= big_lon:
                big_lon = curr_lon
            else:
                small_lon = curr_lon
    # Below checks to unsure that due to +/- longs, patch does not cover whole
    # width but instead is broken into 2 patches, one at the left and one at
    # the right side of the map
    if big_lon - small_lon > 100:
        left_side = [(small_lat, -180.0), (big_lat, -180),
                     (small_lat, small_lon), (big_lat, small_lon)]
        right_side = [(small_lat, big_lon), (big_lat, big_lon),
                      (small_lat, 180), (big_lat, 180)]
        coords = (left_side, right_side)
    else:
        coords = [(small_lat, small_lon), (big_lat, small_lon),
                  (small_lat, big_lon), (big_lat, big_lon)]
    return coords


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
    merged_gdf = clean_geom_col(merged_df, 'geometry')
    return merged_gdf


def clean_geom_col(df, colname):
    '''
    Converts geometry column to usable list of lats/lons
    (instead of one long string)

    Inputs:
        df: a panda dataframe
        colname(str): name of column with geometry information

    Outputs: a geopandas dataframe
    '''
    df[colname] = df[colname].apply(lambda x: \
                                     list(map(float,
                                              re.findall(r'[-|0-9|\.]*[0-9]', x))))
    # Drops obs with invalid coordinates (e.g. latitude = -999)
    df = find_invalid_lats_lons(df, colname)
    # Turns geometry column to list of tuples as (lat, lon) points
    df[colname] = df[colname].apply(lambda x: list(zip(x[1::2], x[::2])))
    # Turns geometry column into polygon shape to plot
    df[colname] = df[colname].apply(geometry.Polygon)
    gdf = gpd.GeoDataFrame(df, geometry=colname)
    # Turns 4 corners into boundaries
    gdf[colname] = gdf[colname].convex_hull
    return gdf


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
    valid_d = {'lats': [0, 90], 'lons': [1, 180]}
    inval_idx = []
    for key in valid_d:
        start, max_val = valid_d[key]
        df[key] = df[col].apply(lambda x: x[start::2])
        df_name = f'{key}_df'
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
        img_name(str):

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
    leg = plt.cm.ScalarMappable(cmap='summer',
                                norm=plt.Normalize(vmin=min(dataframe[colname]),
                                                   vmax=max(dataframe[colname])))
    leg._A = []
    fig.colorbar(leg, cax=cbax, format="%d")
    plt.savefig(img_name)


def find_area(poly_obj):
    '''
    Finds area of polygon shape

    Derived from post here:
    https://gis.stackexchange.com/questions/127607/area-in-km-from-polygon-of-coordinates

    Inputs:
        poly_obj: a polygon (from Shapely lib) object

    Outputs:
        area in square kilometers of the patch
    '''
    geom_area = ops.transform(partial(pyproj.transform,
                                      pyproj.Proj(init='EPSG:4326'),
                                      pyproj.Proj(proj='aea',
                                      lat1=poly_obj.bounds[1],
                                      lat2=poly_obj.bounds[3])), poly_obj)
    return geom_area.area / 1000000


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
            COLS = [x for x in copy.deepcopy(KEYS) if x not in ['latitude',
                                                                'longitude']]
            OUTPUTWRITER.writerow(COLS)
        csvfile.close()
        LAST_FILE = None
    ARGS_LST = []
    FILENAMES_DF = pd.read_csv(ARGS.input_file, dtype='str')
    FILES = list(FILENAMES_DF['filename'])
    if LAST_FILE:
        LAST_IDX = FILES.index(LAST_FILE)
        FILES = FILES[LAST_IDX:]
    for file in FILES[:5]:
        ARGS_LST.append((file, ARGS.outputfile, ARGS.mod02dir, ARGS.mod35dir,
                         ARGS.mod03dir))
    POOL.starmap(make_connecting_csv, ARGS_LST)
    POOL.close()
    POOL.join()
