'''
Katy Koenig

July 2019

Functions to check for invalid hdf files and create graphs of distribution
'''

import os
import sys
import argparse
import csv
import datetime
import glob
import multiprocessing as mp
import pandas as pd
import plotnine as p9
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC


hdf_libdir = '/home/koenig1/scratch-midway2/clouds/src_analysis/lib_hdfs' # change here
sys.path.insert(1, os.path.join(sys.path[0], hdf_libdir))
from alignment_lib import _gen_patches
from alignment_lib import gen_mod35_img
import prg_StatsInvPixel as stats

DATES_FILE = 'clustering_invalid_filelists.txt'
MOD02_DIRECTORY = '/home/koenig1/scratch-midway2/MOD02/clustering'
MOD35_DIRECTORY = '/home/koenig1/scratch-midway2/MOD35/clustering'
OUTPUT_CSV = 'output07102019-2.csv'

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


def create_distributions(csvfile):
    '''
    Creates distribution plots of invalid pixels by patches for each file

    Inputs:
        csvfile(str):

    Outputs:
    '''
    df = pd.read_csv(csvfile)
    grouped = df.groupby('filename') \
                .agg({'patch_no': 'count', 'inval_pixels':'sum'}).reset_index() \
                .rename(columns={'inval_pixels': 'sum_invalid_pixels', 'patch_no': 'patch_count'})
    scatter = p9.ggplot(data=grouped, mapping=p9.aes(x='sum_invalid_pixels', y='patch_count')) \
                        + p9.geom_point(alpha=0.3, color='green') + p9.theme_minimal()

    p9.ggsave(plot=scatter, filename='scatterplot.png')


def get_invalid_info2(file):
    '''
    Searches for desired files and writes csv with each row being a MOD02
    filename, patch number and number of invalid pixels in the patch

    Inputs:
        dates_file(str): txt file with list of desired MOD02
        mod02_dir(str): path where MOD021KM files are located
        mod35_dir(str): path where MOD35_L2 files are locatedf
            dir = '/home/koenig1/scratch-midway2/MOD02/clustering'
            when hdf files located in
            '/home/koenig1/scratch-midway2/MOD02/clustering/clustering_laads_2000_2018_2'
        output_file(str): name of desired output csv

    Outputs:
        None (writes csv)
    '''
    mod02_dir = MOD02_DIRECTORY
    mod35_dir = MOD35_DIRECTORY
    output_file = OUTPUT_CSV

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



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dates_file', type=str, default=DATES_FILE)
    p.add_argument('--processors', type=int, default=5)
    p.add_argument('--outputfile', type=str, default=OUTPUT_CSV)
    args = p.parse_args()
    start_time = datetime.datetime.now()
    print(start_time)
    print(args.processors)
    
    #Initializes pooling process for parallelization
    pool = mp.Pool(processes=args.processors)

    # If output csv exits, assumes cutoff by RCC so deletes last entry and appends only new dates
    if os.path.exists(args.outputfile):
        print('Checking for completion')
        completed = pd.read_csv(args.outputfile)
        last_file = completed.tail(1)['filename'].tolist()[0]
        done = completed[completed['filename'] != last_file]
        print('Writing updated csv')
        done.to_csv(args.outputfile)
    else:
        print(datetime.datetime.now())
        # Initializes output csv to be appended later
        with open(args.outputfile, 'w') as csvfile:
            outputwriter = csv.writer(csvfile, delimiter=',')
            outputwriter.writerow(['filename', 'patch_no', 'inval_pixels'])
        csvfile.close()
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
