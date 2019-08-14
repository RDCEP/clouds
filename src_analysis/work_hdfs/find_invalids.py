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
from pyhdf.SD import SD, SDC
import geolocation
import map_cluster as mc


hdf_libdir = '/home/koenig1/scratch-midway2/clouds/src_analysis/lib_hdfs' # change here
sys.path.insert(1, os.path.join(sys.path[0], hdf_libdir))
from alignment_lib import _gen_patches
from alignment_lib import gen_mod35_img
import prg_StatsInvPixel as stats

DATES_FILE = 'clustering_invalid_filelists.txt'
MOD02_DIRECTORY = '/home/koenig1/scratch-midway2/MOD02/clustering'
MOD35_DIRECTORY = '/home/koenig1/scratch-midway2/MOD35/clustering'
OUTPUT_CSV = 'output07102019.csv'
MAIN_DIR = '/home/koenig1/clouds/src_analysis/combined/test'


def get_invalid_info(dates_file=DATES_FILE, mod02_dir=MOD02_DIRECTORY,
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
    with open(output_file, 'w') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        outputwriter.writerow(['filename', 'patch_no', 'inval_pixels'])
    csvfile.close()
    # Finds name of desired MOD02 hdf files to be analyzed
    with open(dates_file, "r") as file:
        dates = file.readlines()
    desired_files = dates[0].replace('hdf', 'hdf ').split()
    for file in desired_files:
        # Finds actual MOD02 files
        mod02_path = glob.glob(mod02_dir + '/*/' + file)[0]
        bname = os.path.basename(file)
        date = bname[10:22]
        # Finds corresponding MOD35
        mod35_path = glob.glob(mod35_dir + '/*/*' + date + '*.hdf')[0]
        fillvalue_list, mod02_img = stats.gen_mod02_img(mod02_path)
        hdf_m35 = SD(mod35_path, SDC.READ)
        clouds_mask_img = stats.gen_mod35_img(hdf_m35)
        mod02_patches = _gen_patches(mod02_img, normalization=False)
        # Checks validity of pixels for each file and writes csv
        stats.check_invalid_clouds2(output_file, file, mod02_patches,
                                    clouds_mask_img, fillvalue_list, thres=0.3)


def get_invalid_info2(file, mod02_dir=MOD02_DIRECTORY,
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
        mod02_dir(str):
        mod35_dir(str):
        output_file(str):

    Outputs:
        None (writes csv)
    '''
    # Finds actual MOD02 file
    mod02_glob = glob.glob(mod02_dir + '/*/*/' + file)
    mod02_patches, fillvalue_list = gen_mod02(mod02_glob, file)
    # Finds corresponding MOD35
    date = os.path.basename(file)[10:22]
    mod35_glob = glob.glob(mod35_dir + '/*/*' + date + '*.hdf')
    clouds_mask_img = gen_mod35(mod35_glob)
    if clouds_mask_img and mod02_patches:
        # Checks validity of pixels for each file and writes csv
        stats.check_invalid_clouds2(output_file, file, mod02_patches,
                                    clouds_mask_img, fillvalue_list, thres=0.3)


def get_info_for_location(mod02_dir, mod35_dir, mod03_dir):
    '''
    '''
    mod02_files = glob.glob(mod02_dir + '/*/*/*/*.hdf')
    for mod02_file in mod02_files:
        mod02_patches, fillvalue_list = gen_mod02(mod02_file)
        mod02_file[0][50:]

    latitude, longitude = mc.gen_mod03(mod03_path)


def gen_mod02(mod02_file, file):
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
        print("No mod02 file downloaded for " + file)
        return None
    elif len(mod02_file) == 1:
        mod02_file = mod02_file[0]
    fillvalue_list, mod02_img = stats.gen_mod02_img(mod02_file)
    mod02_patches = _gen_patches(mod02_img, normalization=False)
    return mod02_patches, fillvalue_list


def gen_mod35(mod35_file, date):
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
        print("No mod35 file downloaded for " + date)
        return None
    elif len(mod35_file) == 1:
        mod35_file = mod35_file[0]
    hdf_m35 = SD(mod35_file, SDC.READ)
    clouds_mask_img = stats.gen_mod35_img(hdf_m35)
    return clouds_mask_img


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
        print("No mod03 file downloaded for " + date)
        return None
    elif len(mod03_file) == 1:
        mod03_file = mod03[0]
    mod03_hdf = SD(mod03_file, SDC.READ)
    lat = mod03_hdf.select('Latitude')
    latitude = lat[:, :]
    lon = mod03_hdf.select('Longitude')
    longitude = lon[:, :]
    return latitude, longitude


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dates_file', type=str, default=DATES_FILE)
    p.add_argument('--processors', type=int, default=5)
    p.add_argument('--outputfile', type=str, default=OUTPUT_CSV)
    args = p.parse_args()
    print("Num. Processors: " + args.processors)
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
        print('Writing updated csv w/ only completed MOD02 files')
        done.to_csv(args.outputfile, index=False)
    else:
        # Initializes output csv to be appended later
        with open(args.outputfile, 'w') as csvfile:
            outputwriter = csv.writer(csvfile, delimiter=',')
            outputwriter.writerow(['filename', 'patch_no', 'inval_pixels'])
        csvfile.close()
        last_file = None
    # Finds name of desired MOD02 hdf files to be analyzed
    with open(args.dates_file, "r") as file:
        dates = file.readlines()
    desired_files = dates[0].replace('hdf', 'hdf ').split()
    if last_file:
        last_idx = desired_files.index(last_file)
        desired_files = desired_files[last_idx:]

    pool.map(get_invalid_info2, desired_files)
    pool.close()
    pool.join()

