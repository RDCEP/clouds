'''
Summer 2019

Katy Koenig

Functions to download entire world MODIS data (non location-specific)

(To download location specific data, check out the api-requests directory)

'''
import os
import argparse
import datetime
import urllib
import csv
from urllib.request import urlopen
import multiprocessing as mp
from bs4 import BeautifulSoup
import requests
import pandas as pd
import prg_gen_rndm_metadata as pgrm


BASE_URL = {'MOD02': 'https://ladsweb.modaps.eosdis.nasa.gov/archives/allData/61/MOD021KM',
            'MOD35': 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD35_L2', 
            'MOD06': 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD06_L2',
            'MOD03': 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD03'}


def get_href_lists2(url, keyword):
    '''
    Finds relevant hrefs from a given website

    Inputs:
        url: a url
        keyward: string w/ keyword for relevant hrefs

    Outputs: list of relevant hrefs
    '''
    r = requests.get(url)
    https = r.text.encode('iso-8859-1')
    bsobj = BeautifulSoup(https, 'html5lib')
    link_tags = bsobj.findAll("a")
    hrefs = []
    for link in link_tags:
        if link.has_attr('href'):
            raw_site = link['href']
            if keyword in raw_site and ".hdf" in raw_site:
                hrefs.append(raw_site)
    return hrefs


def bool_fsize2(url, thresval=1):
    '''
    Checks that webpage is larger than a certain threshold

    Inputs:
        url: url for website
        thresval: threshold value

    Outputs: boolean
    '''
    coef = 1/1000/1000
    r = requests.get(url)
    fsize = len(r.content) * coef
    return fsize >= thresval


def download_data(url, start_time, outputdir='.'):
    '''
    Downloads data/images

    Inputs:
        url(str): website path
        start_time(time obj):
        outputdir(str): folder in which downloaded objected save

    Outputs: prints notice of completed downloads & time from start_time
             each was downloaded
    '''
    filename = os.path.basename(url)
    urllib.request.urlretrieve(url, os.path.join(outputdir, filename))
    print("## FINISH DOWNLOAD ! ## %s" % filename)
    curr_time = datetime.datetime.now()
    print(pd.Timedelta(curr_time - start_time))


def delta_day(year='0000', month='00', day='00'):
    '''
    Return # days of input date by the scale of 365
    e.g. 2000/1/10 = 10, 2000/12/31 = 365

    Inputs:
        year(str): number representing year
        month(str): number of month into year
        day(str): day of month

    Outputs: integer of number of days
    '''
    _year = int(year)
    _month = int(month)
    _day = int(day)
    delta = datetime.date(_year, _month, _day) - datetime.date(_year, 1, 1)
    dday = delta.days+1
    if dday < 10:
        delta_day = "00" + str(dday)
    elif 100 > dday >= 10:
        delta_day = "0" + str(dday)
    elif 1000 > dday >= 100:
        delta_day = str(dday)
    return delta_day


def combining_fn(iline, url, thresval, outputdir, start_time):
    '''
    Download requested HDF files into specified directory

    Inputs:
        iline(datetime obj): date for data requested
        base_url(str): url base
        keyward(str): keyword to find relevant images
        thresval(int): value for threshold
        outputdir(str): folder in which downloaded objected save

    Outputs: saved HDF files
    '''
    date = iline.split('\n')[0]
    year = date[:4]
    days = delta_day(year=date[:4], month=date[5:7], day=date[8:])
    url = f'{url}/{year}/{days}/'
    # Check if valid url
    response = requests.get(url)
    if response.status_code == 200:
        # href_lists
        href_list = get_href_lists2(url, args.keyward)
        for ihref in href_list:
            https = url + os.path.basename(ihref)
            bfsize = bool_fsize2(https, thresval)
            if bfsize:
                response = requests.get(https)
                if response.status_code == 200:
                    download_data(https, start_time, outputdir)
                else:
                    href_list.append(ihref)
    else:
        # Adds dates w/o data available to running list of dates
        with open('no-data-dates.txt', 'a') as f:
            f.write(str(date) + "\n")
        # Note: duplicates are allowed on list--need to remove duplicates
        # periodically


def download_from_name(file, keyword, outputdir, start_time):
    '''
    Downloads files based on the desired date/time of the hdf file.
    This is used for downloaded a corresponding file, e.g. you have the MOD02
    file downloaded (and it's name) and want the corresponding MOD35 file.

    Inputs:
        file(str): name of desired file with only year, date and time info
            e.g. '2011169.0420'
        keyword(str): name of desired product
            (see keys of BASE_URL dictionary above for options)
        outputdir(str): name of directory in which to store files

    Outputs: saved HDF files
    '''
    diff = 12 - len(file)
    file = file + '0' * diff
    base_url = BASE_URL[keyword]
    year = file[0:4]
    print(file)
    date = file[4:7]
    time = file[8:12]
    url = f'{base_url}/{year}/{date}/'
    response = requests.get(url)
    if response.status_code == 200:
        # Gets specific href for only 1 time (not all images on a page)
        href_list = get_href_lists2(url, '.' + time)
        for ihref in href_list:
            https = url + os.path.basename(ihref)
            response = requests.get(https)
            if response.status_code == 200:
                download_data(https, start_time, outputdir)
            else:
                href_list.append(ihref)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--download_process', type=str, default='combining_fn')
    p.add_argument('--outputdir', type=str, default=os.getcwd())
    p.add_argument('--processors', type=int, default=20)
    p.add_argument('--keyword', type=str, default='MOD35')
    p.add_argument('--input_csv', type=str, default="/scratch/midway2" + \
                   "/koenig1/clouds/src_analysis/work_hdfs/missing_mod03.csv")
    p.add_argument('--thresval', type=int, default=100)
    p.add_argument('--days', type=int, default=1)
    p.add_argument('--start', type=str, default='2000-02-24')
    p.add_argument('--end', type=str, default='2019-03-27')
    p.add_argument('--datedata', type=str, default=None)
    args = p.parse_args()
    start_time = datetime.datetime.now()
    print(f"time process has began: {start_time}")
    print(f"num of processors used : {args.processors}")
    # Set up pool for multiprocessing/parallelization
    pool = mp.Pool(processes=args.processors)
    args_lst = []
    # Options to download: combining_fn or download_from_name (see above fns)
    if args.download_process == 'combining_fn':
        base_url = BASE_URL[args.keyword]
        # To download specific dates
        if args.datedata:
            with open(args.datedata, 'r') as ifile:
                for iline in ifile.readlines():
                    args_lst.append((iline, base_url, args.thresval,
                                     args.outputdir, start_time))
        # To generate random dates between two specified dates
        else:
            datedata = pgrm.gen_random_date(ndays=args.days, stime=args.start,
                                            etime=args.end)
            pgrm.save_filelist(datedata, args.outputdir,
                               oname=f'dates_created {datetime.datetime.today()}'
                               .strftime('%Y-%m-%d'))
            for iline in datedata:
                args_lst.append((iline, base_url, args.thresval, args.outputdir,
                                 start_time))       
    # second option: download specific dates/times
    if args.download_process == 'download_from_name':
        files_df = pd.read_csv(args.input_csv, dtype='str')
        # finds desired dates/times to be download & avoids duplicates
        file_set = set(files_df['filename'].tolist())
        for file in file_set:
            args_lst.append((str(file), args.keyword, args.outputdir,
                             start_time))
    pool.starmap_async(args.download_process, args_lst)
    pool.close()
    pool.join()
