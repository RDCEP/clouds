'''
Katy Koenig

July 2019

Functions to check for invalid hdf files
'''

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
from alignment_lib import gen_mod02_img_sigle, gen_mod35_img_single
from alignment_lib import mod02_proc_sds_single
from alignment_lib import _gen_patches
from alignment_lib import const_clouds_array


DATES_FILE = 'clustering_invalid_filelists.txt'
HDF_DIRECTORY = '/home/koenig1/scratch-midway2/MOD02/clustering'

def get_dates(filename=DATES_FILE, directory=HDF_DIRECTORY):
    '''

    Inputs:
        filename:

    Outputs:
    '''
    with open(filename, "r") as file:
        dates = file.readlines()
    desired_files = dates[0].replace('hdf', 'hdf ').split()

    for path, directories, files in os.walk(directory):
        hdf_lst = [x for x in files]


    for date in desired_files:
        os.listdir(directory)


