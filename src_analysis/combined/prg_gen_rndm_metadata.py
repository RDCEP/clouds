'''
Functions to Generate Dates
_*_ coding : utf-8 _*_
'''

import random
import time
import pandas as pd

UNUSABLE_DATES = ['no-data-dates.txt', 'mod02_training_dates.txt', 'pv_ecal.csv']

def get_proptime(stime='2000-02-24', etime='2019-03-27', _format='%Y-%m-%d',
                 prop=1.0):
    '''
    prop specifies how a proportion of the intervalto be taken after start.
    Range is between 0-1(float).
    The returned time will be in the specified format.
    '''
    sdate = time.mktime(time.strptime(stime, _format))
    edate = time.mktime(time.strptime(etime, _format))
    ptime = sdate + prop*(edate - sdate)
    return time.strftime(_format, time.localtime(ptime))


def get_bad_dates(file_lst=UNUSABLE_DATES):
    '''
    Returns list of dates from a text file

    Input:
        file_lst(list): list of strings of files with the dates NOT to be
                        downloaded

    Outputs: list of dates as strings
    '''
    dfs_to_concat = []
    for file in file_lst:
        if 'txt' in file:
            ind_df = pd.read_fwf(file, header=None)
        if 'csv' in file:
            ind_df = pd.read_csv(file, header=None)
            ind_df[0] = ind_df[0].map(lambda x: x[5:15])
            ind_df = ind_df[0].to_frame()
        dfs_to_concat.append(ind_df)
    df = pd.concat(dfs_to_concat)
    df.rename(columns={0: 'date'}, inplace=True)
    df.to_csv('total_bad_dates.csv')
    return list(df['date'])


def gen_random_date(ndays=1, stime='2000-02-24', etime='2019-03-27',
                    _format='%Y-%m-%d', rand_seed=123456):
    '''
    Generates random date(s) between two given dates

    Inputs:
        ndays(int): number of days to be generates
        stime(str): lower-bound date
        etime(str): upper-bound date
        _format(str): format for dates
        rand_seed(int): seed for setting random.random()

    Output: list of dates
    '''
    filelist = []
    missing_bad_list = get_bad_dates() # avoid missing data days
    random.seed(rand_seed)
    while len(filelist) < ndays:
        ctime = get_proptime(stime=stime, etime=etime, _format=_format,
                             prop=random.random())
        if not ctime in missing_date_list:
            filelist.append(ctime)
    filelist.sort()
    return filelist


def save_filelist(filelist=[], outdir='./', oname='output'):
    '''
    Saves list of dates of dates to be downloaded

    Inputs:
        filelist: list of files
        outdir(str): directory in which list of files should be downloaded
        oname(str): name for file created

    Outputs: Prints name of 
    '''
    with open(outdir+"/"+oname+".txt", 'w') as ofile:
        for iline in filelist:
            ofile.write(iline+'\n')
    print(" ### File Saved %s/%s.txt " % (outdir, oname))
