'''
Katy Koenig

July/August 2019

Functions to check for invalid hdf files and get stats regarding invalids
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
MOD02_DIRECTORY = ''
MOD35_DIRECTORY = ''
OUTPUT_CSV = ''
MAIN_DIR = ''


def get_invalid_info(dates_file=INPUT_FILE, mod02_dir=MOD02_DIRECTORY,
                     mod35_dir=MOD35_DIRECTORY, output_file=OUTPUT_CSV):
    '''
    Searches for desired files and writes csv with each row being a MOD02
    filename, patch number and number of invalid pixels in the patch

    Inputs:
        dates_file(str): txt file with list of desired MOD02
        mod02_dir(str): path where MOD021KM files are located
        mod35_dir(str): path where MOD35_L2 files are located
        Note for both mod02_dir & mod35_dir:
            dir = '/home/koenig1/scratch-midway2/MOD02/clustering'
            when hdf files located in
            '/home/koenig1/scratch-midway2/MOD02/clustering/clustering_laads_2000_2018_2'
        output_file(str): name of desired output csv

    Outputs:
        None (writes csv)
    '''
    # Initializes output csv to be appended later
    with open(output_file, 'w') as output_csv:
        outputwriter = csv.writer(output_csv, delimiter=',')
        outputwriter.writerow(['filename', 'patch_no', 'inval_pixels'])
    output_csv.close()
    # Finds name of desired MOD02 hdf files to be analyzed
    with open(dates_file, "r") as desired_dates:
        dates = desired_dates.readlines()
    desired = dates[0].replace('hdf', 'hdf ').split()
    for mod02_file in desired:
        # Finds actual MOD02 files
        mod02_path = glob.glob(f'{mod02_dir}/*/{mod02_file}')[0]
        bname = os.path.basename(mod02_file)
        date = bname[10:22]
        # Finds corresponding MOD35
        mod35_path = glob.glob(f'{mod35_dir}/*/*{date}*.hdf')[0]
        fillvalue_list, mod02_img = stats.gen_mod02_img(mod02_path)
        clouds_mask_img = stats.gen_mod35_img(SD(mod35_path, SDC.READ))
        mod02_patches = _gen_patches(mod02_img, normalization=False)
        # Checks validity of pixels for each file and writes csv
        stats.check_invalid_clouds2(output_file, mod02_file, mod02_patches,
                                    clouds_mask_img, fillvalue_list, thres=0.3)


def get_invalid_info2(filename, mod02_dir=MOD02_DIRECTORY,
                      mod35_dir=MOD35_DIRECTORY, output_file=OUTPUT_CSV):
    '''
    Searches for desired files and writes csv with each row being a MOD02
    filename, patch number and number of invalid pixels in the patch

    Note the difference between this function and the one above:
    get_invalid_info() is designed to run in ipython3 while
    get_invalid_info2() is optimized to run on the command line
    using the multiprocessing library for task parallelization

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


def get_info_for_location(mod02_dir, mod35_dir, mod03_dir, outputfile, nparts):
    '''
    Creates csv with invalid pixel information and location for files of
    specific locations

    Inputs:
        mod02_dir(str): directory in which MOD02 hdf files are saved
        mod35_dir(str): directory in which MOD35 hdf files are saved
        mod03_dir(str): directory in which MOD03 hdf files are saved
        outputfile(str): desired name of output csv
        nparts(int): number of partitions/cores to be used for parallelization

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
        location_date = mod02_file[50:70]
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
                         agg({'patch_no': 'count', 'inval_pixels': 'sum'}). \
                         rename(columns={'patch_no': 'patch_count'}).reset_index()
    by_loc['pct_inval'] = by_loc['inval_pixels'] / by_loc['patch_count'] * 100
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
    print(mod03_file)
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
    P.add_argument('--input_file', type=str, default=INPUT)
    P.add_argument('--processors', type=int, default=5)
    P.add_argument('--outputfile', type=str, default=OUTPUT_CSV)
    ARGS = P.parse_args()
    print(f"Num. Processors: {ARGS.processors}")
    #Initializes pooling process for parallelization
    POOL = mp.Pool(processes=ARGS.processors)
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
    pritn(DESIRED_FILES)

    # POOL.map(get_invalid_info2, DESIRED_FILES)
    # POOL.close()
    # POOL.join()
