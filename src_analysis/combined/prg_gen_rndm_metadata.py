'''
Functions to Generate Dates
_*_ coding : utf-8 _*_
'''

import random
import time
import pandas as pd


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


def get_missing_dates(filename='no-data-dates.txt'):
    '''
    Returns list of dates from a text file

    Input:
    filename(str): filename with dates

    Outputs: list of dates as strings
    '''
    df = pd.read_fwf(filename, header=None)
    df.rename(columns={0: 'date'}, inplace=True)
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
    missing_date_list = get_missing_dates() # avoid missing data days
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
