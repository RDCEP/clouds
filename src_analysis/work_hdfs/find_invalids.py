'''
Katy Koenig

July/August 2019

Functions to check hdf files and get stats regarding invalids pixels
'''

import os
import sys
import argparse
import csv
import glob
import multiprocessing as mp
import pandas as pd
import numpy as np
from pyhdf.SD import SD, SDC
import geolocation as geo
import prg_StatsInvPixel as stats

HDF_LIBDIR = '/home/koenig1/scratch-midway2/clouds/src_analysis/lib_hdfs' # change here
sys.path.insert(1, os.path.join(sys.path[0], HDF_LIBDIR))
from analysis_lib import _gen_patches

# Put in your corresponding file/directories below
INPUT_FILE = 'clustering_invalid_filelists.txt'
MOD02_DIRECTORY = '/home/koenig1/scratch-midway2/MOD021KM'
MOD03_DIRECTORY = '/home/koenig1/scratch-midway2/MOD03'
MOD35_DIRECTORY = '/home/koenig1/scratch-midway2/MOD35_L2'
OUTPUT_CSV = 'output.csv'


def get_invalid_info(filename, mod02_dir, mod35_dir, output_file):
    '''
    Searches for desired files and writes csv with each row being a MOD02
    filename, patch number and number of invalid pixels in the patch

    Inputs:
        file(str): name of mod02 hdf file
        mod02_dir(str): directory in which MOD02 hdf file is saved
        mod35_dir(str): directory in whichg MOD35 hdf file is saved
        output_file(str): desired output csv file name

    Outputs:
        None (writes csv)
    '''
    # Finds actual MOD02 file
    mod02_glob = glob.glob(f'{mod02_dir}/*/*/{filename}')
    mod02_patches, fillvalue_list = gen_mod02(mod02_glob, filename)
    # Finds corresponding MOD35
    date = os.path.basename(filename)[10:22]
    mod35_glob = glob.glob(f'{mod35_dir}/*/*{date}*.hdf')
    clouds_mask_img = gen_mod35(mod35_glob)
    if isinstance(clouds_mask_img, np.ndarray) and isinstance(mod02_patches,
                                                              np.ndarray):
        # Checks validity of pixels for each file and writes csv
        stats.check_invalid_clouds2(output_file, filename, mod02_patches,
                                    clouds_mask_img, fillvalue_list, thres=0.3)


def get_info_for_location(mod02_dir, mod35_dir, mod03_dir, outputfile,
                          nparts=4):
    '''
    Creates csv with invalid pixel information and location for files of
    specific locations

    Inputs:
        mod02_dir(str): directory in which MOD02 hdf files are saved
        mod35_dir(str): directory in which MOD35 hdf files are saved
        mod03_dir(str): directory in which MOD03 hdf files are saved
        outputfile(str): desired name of output csv
        nparts(int): number of partitions/cores to be used for parallelization

    Note that this function assumes a that path for any MOD file is stored
    using the following format:
    mod02/clustering/location/date/MODfile.hdf
    (This is how api_requests.py saves files)

    Outputs: None (creates and saves a csv file)
    '''
    with open(outputfile, 'w') as out_csv:
        outputwriter = csv.writer(out_csv, delimiter=',')
        outputwriter.writerow(['filename', 'patch_no', 'invalid_pixels',
                               'geometry'])
    out_csv.close()
    mod02_files = glob.glob(f'{mod02_dir}/*/*/*/*.hdf')
    for mod02_file in mod02_files:
        file_base = mod02_file[-34:-22]
        location_date = mod02_file[-63:-45]
        mod35_path = glob.glob(f'{mod35_dir}/*/{location_date}/*{file_base}*.hdf')
        cloud_mask_img = gen_mod35(mod35_path, file_base)
        mod03_path = glob.glob(f'{mod03_dir}/*/{location_date}/*{file_base}*.hdf')
        latitude, longitude = gen_mod03(mod03_path, file_base)
        try:
            patches, fill_values, lats, longs = gen_mod02(mod02_file, file_base,
                                                          latitude, longitude)
            if isinstance(lats, np.ndarray) and isinstance(cloud_mask_img,
                                                           np.ndarray):
                geo.connect_geolocation(f'{file_base}_{location_date}',
                                        outputfile, patches, fill_values, lats,
                                        longs, cloud_mask_img, nparts)
        except:
            print(f'Missing Files - Unable to write info for {mod02_file}')
            pass


def get_stats_spec_locs(spec_loc_csv):
    '''
    Generates statistics regarding invalid pixels for specific locations,
    specifically to check if areas that have a high absolute invalid
    pixel count also have a high relative invalid pixel, i.e. do areas with
    a lot of invalid pixels also have a lot of patches?

    Inputs:
        spec_loc_csv(str): csv with invalid pixel info for specific locations

    Outputs:
        spec_loc_df: a pandas dataframe with each row as an individual patch
        by_loc: a pandas dataframe that is grouped by location
    '''
    types_d = {'filename': 'str', 'patch_no': 'int', 'invalid_pixels': 'int',
               'geometry': 'str'}
    spec_loc_df = pd.read_csv(spec_loc_csv, dtype=types_d)
    spec_loc_df['location'] = spec_loc_df['filename'].apply(lambda x: x[13:-11])
    by_loc = spec_loc_df.groupby('location'). \
                         agg({'patch_no': 'count', 'invalid_pixels': 'sum'}). \
                         rename(columns={'patch_no': 'patch_count'}).reset_index()
    by_loc['invals_per_patch'] = by_loc['invalid_pixels'] / by_loc['patch_count']
    return spec_loc_df, by_loc


def gen_mod02(mod02_file, filename, latitude=None, longitude=None):
    '''
    Generates the mod02 patches and the fill value list for a MOD02 hdf file

    Inputs:
        mod02_file(str or lst of 1 item): either a mod02 filename or a list of
            mod02 files as found using glob.glob()
        file(str): mod02 hdf file name

    Outputs:
        mod02_patches: numpy array of arrays representing MOD02 patches
        fillvalue_list: list of integers for each fill value
    '''
    if not mod02_file:
        print(f"No mod02 file downloaded for {filename}")
        return None
    if len(mod02_file) == 1:
        mod02_file = mod02_file[0]
    return geo.make_patches(mod02_file, latitude, longitude)


def gen_mod35(mod35_file, date=None):
    '''
    Generate image of cloud masks if MOD35 hdf file available

    Inputs:
        mod35_file(str or lst of 1 item): either a mod35 filename or a list of
            mod35 files as found using glob.glob()
        date(str): year and day of year for a given file

    Outputs: cloud mask image (numpy array created from MOD35 image)
             or None if hdf file not available
    '''
    if not mod35_file:
        print(f"No mod35 file downloaded for {date}")
        return None
    if len(mod35_file) == 1:
        mod35_file = mod35_file[0]
    try:
        hdf_m35 = SD(mod35_file, SDC.READ)
        clouds_mask_img = stats.gen_mod35_img(hdf_m35)
        return clouds_mask_img
    except:
        print(f'Issue with downloaded MOD35 file {mod35_file}')
        pass


def gen_mod03(mod03_file, date):
    '''
    Reads in MOD03 hdf file and converts data to relevant latitude and
    longitude arrays

    Inputs:
        mod03_file(str or lst of 1 item): either a mod03 filename or a list of
            mod03 files as found using glob.glob()
        date(str): year and day of year for a given file

    Outputs:
        latitude: numpy array of arrays with the latitudes of each pixel
        longitude: numpy array of arrays with the longitudes of each pixel
    '''
    if not mod03_file:
        print(f"No mod03 file downloaded for {date}")
        return None
    if len(mod03_file) == 1:
        mod03_file = mod03_file[0]
    try:
        mod03_hdf = SD(mod03_file, SDC.READ)
        lat = mod03_hdf.select('Latitude')
        latitude = lat[:, :]
        lon = mod03_hdf.select('Longitude')
        longitude = lon[:, :]
        return latitude, longitude
    except:
        print(f'Issue with downloaded MOD03 file {mod03_file}')
        pass


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument('--process', type='str', default=None)
    P.add_argument('--input', type='str', default=INPUT)
    P.add_argument('--mod02_dir', type='str', default=MOD02_DIRECTORY)
    P.add_argument('--mod35_dir', type='str', default=MOD35_DIRECTORY)
    P.add_argument('--mod03_dir', type='str', default=MOD03_DIRECTORY)
    P.add_argument('--outputfile', type='str', default='')
    P.add_argument('--processors', type='str', default=4)
    ARGS = P.parse_args()
    #Initializes pooling process for parallelization
    if ARGS.process == 'get_invalid_info':
        POOL = mp.Pool(processes=ARGS.processors)
        args_lst = []
        # If output csv exits, assumes cutoff by RCC so deletes last entry
        # and appends only new dates
        if os.path.exists(ARGS.outputfile):
            print('Checking for completion')
            COMPLETED = pd.read_csv(ARGS.outputfile)
            COMPLETED = COMPLETED[COMPLETED['filename'].notnull()]
            LAST_FILE = COMPLETED.tail(1)['filename'].tolist()[0]
            DONE = COMPLETED[COMPLETED['filename'] != LAST_FILE]
            print('Writing updated csv w/ only COMPLETED MOD02 files')
            DONE.to_csv(ARGS.outputfile, index=False)
        else:
            # Initializes output csv to be appended later
            with open(ARGS.outputfile, 'w') as csvfile:
                OUTPUTWRITER = csv.writer(csvfile, delimiter=',')
                OUTPUTWRITER.writerow(['filename', 'patch_no', 'inval_pixels'])
            csvfile.close()
            LAST_FILE = None
        if 'txt' in ARGS.input:
        # Finds name of desired MOD02 hdf files to be analyzed
            with open(ARGS.input, "r") as file:
                DATES = file.readlines()
            DESIRED_FILES = DATES[0].replace('hdf', 'hdf ').split()
            if LAST_FILE:
                LAST_IDX = DESIRED_FILES.index(LAST_FILE)
                DESIRED_FILES = DESIRED_FILES[LAST_IDX:]
        else:
            DESIRED_FILES = os.listdir(ARGS.input)
        for file in DESIRED_FILES:
            args_lst.append((file, ARGS.mod02_dir, ARGS.mod35_dir,
                             ARGS.output_file))
        POOL.starmap_async(get_invalid_info, args_lst)
        POOL.close()
        POOL.join()

    if ARGS.process == 'get_info_for_location':
        get_info_for_location(ARGS.mod02_dir, ARGS.mod35_dir, ARGS.mod03_dir,
                              ARGS.outputfile, ARGS.processors)
