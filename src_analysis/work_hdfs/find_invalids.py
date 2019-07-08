'''
Katy Koenig

July 2019

Functions to check for invalid hdf files
'''

import os


DATES_FILE = 'clustering_invalid_filelists.txt'
HDF_DIRECTORY = '/home/koenig1/scratch-midway2/MOD02/clustering'
DEST_DIRECTORY = '/home/koenig1/scratch-midway2/distribution'

def get_dates(filename=DATES_FILE, directory=HDF_DIRECTORY, destination=DEST_DIRECTORY):
    '''
    Searches for desired files and links them to destination directory to be called later

    Inputs:
        filename:
        directory:
        destination:

    Outputs:
        None
    '''
    with open(filename, "r") as file:
        dates = file.readlines()
    desired_files = dates[0].replace('hdf', 'hdf ').split()

    for path, directories, files in os.walk(directory):
        for file in files:
            if file in desired_files:
                os.link(path + "/" + file, destination)

