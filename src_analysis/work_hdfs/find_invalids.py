'''
Katy Koenig

July 2019

Functions to check for invalid hdf files and create graphs of distribution
'''

import os
import sys
import glob
from pyhdf.SD import SD, SDC

hdf_libdir = '/home/koenig1/scratch-midway2/clouds/src_analysis/lib_hdfs' # change here
sys.path.insert(1,os.path.join(sys.path[0],hdf_libdir))
from alignment_lib import _gen_patches
from alignment_lib import gen_mod35_img
import prg_StatsInvPixel as stats

DATES_FILE = 'test.txt'
MOD02_DIRECTORY = '/home/koenig1/scratch-midway2/MOD02/clustering'
MOD35_DIRECTORY = '/home/koenig1/scratch-midway2/MOD35/clustering'
DEST_DIRECTORY = '/home/koenig1/scratch-midway2/clouds/src_analysis/work_hdfs/distribution'

def get_dates(dates_file=DATES_FILE, mod02_dir=MOD02_DIRECTORY, mod35_dir=MOD35_DIRECTORY, destination=DEST_DIRECTORY, output_file='output.csv'):
    '''
    Searches for desired files and links them to destination directory to be called later

    Inputs:
        filename(str):
        mod02_dir(str):
        mod35_dir(str):
        destination(str):

    Outputs:
        None
    '''
    with open(output_file, 'w') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        outputwriter.writerow(['filename', 'patch_no', 'inval_pixels'])
    csvfile.close()
    with open(dates_file, "r") as file:
        dates = file.readlines()
    desired_files = dates[0].replace('hdf', 'hdf ').split()
    for file in desired_files:
        mod02_path = glob.glob(mod02_dir + '/*/' + file)[0]
        #os.link(mod02_path, destination)
        bname = os.path.basename(file)
        date = bname[10:22]
        mod35_path = glob.glob(mod35_dir + '/*/*' + date + '*.hdf')[0]
        fillvalue_list, mod02_img = stats.gen_mod02_img(mod02_path)
        hdf_m35 = SD(mod35_path, SDC.READ)
        clouds_mask_img = stats.gen_mod35_img(hdf_m35)
        mod02_patches = _gen_patches(mod02_img, normalization=False)
        stats.check_invalid_clouds2(output_file, file, mod02_patches, clouds_mask_img, fillvalue_list, thres=0.3)

