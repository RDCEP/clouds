'''
Functions to download data
 _*_ coding : utf-8 _*_
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


def get_href_lists2(url, keyward="MOD021KM.A"):
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
            if keyward in raw_site and ".hdf" in raw_site:
                #print(raw_site)
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
    fsize = len(r.content)*coef
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
        delta_day = "00"+str(dday)
    elif 100 > dday >= 10:
        delta_day = "0"+str(dday)
    elif 1000 > dday >= 100:
        delta_day = str(dday)
    return delta_day


def combining_fn(iline, url, thresval, outputdir, start_time):
    '''
    Download requests HDF files into specified directory

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
    url = args.url+'/'+year+'/'+days+'/'
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
        # Note: duplicates are allowed on list--remove duplicates periodically


def download_mod03(csv_file='inval_files.csv',
                   url='https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD03/',
                   keyword='MOD03', outputdir='/home/koenig1/scratch-midway2/MOD03/clustering/invalid_pixels', thresval=1):
    start_time = datetime.datetime.now()
    with open(csv_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            year = row[1][:7]
            date = row[1][8:]
            url = baseurl +'/'+year+'/'+days+'/'
            response = requests.get(url)
            if response.status_code == 200:
                # href_lists
                href_list = get_href_lists2(url, keyword)
                for ihref in href_list:
                    https = url + os.path.basename(ihref)
                    bfsize = bool_fsize2(https, thresval)
                    print(bfsize)
                    if bfsize:
                        response = requests.get(https)
                        if response.status_code == 200:
                            download_data(https, start_time, outputdir)
                        else:
                            href_list.append(ihref)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--url', help=' base url without datenumber and year',
                   type=str,
                   default="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD35_L2")
    p.add_argument('--outputdir', type=str, default=os.getcwd())
    p.add_argument('--keyward', type=str, default='MOD35_L2.A')
    p.add_argument('--thresval', type=int, default=100)
    p.add_argument('--processors', type=int, default=mp.cpu_count() - 1)
    p.add_argument('--days', type=int, default=1)
    p.add_argument('--start', type=str, default='2000-02-24')
    p.add_argument('--end', type=str, default='2019-03-27')
    p.add_argument('--datedata', type=str, default=None)

    args = p.parse_args()
    os.makedirs(args.outputdir, exist_ok=True)
    start_time = datetime.datetime.now()
    print(start_time)
    print(args.processors)
    #Initializes pooling process for parallelization
    pool = mp.Pool(processes=args.processors)
    # Loads date metadata and creates arg tuple for each worker in pool
    args_lst = []
    if args.datedata:
        with open(args.datedata, 'r') as ifile:
            for iline in ifile.readlines():
                args_lst.append((iline, args.url, args.thresval, args.outputdir, start_time))
    else:
        datedata = pgrm.gen_random_date(ndays=args.days, stime=args.start, etime=args.end)
        pgrm.save_filelist(datedata, args.outputdir, oname='dates_created ' + datetime.datetime.today()
                           .strftime('%Y-%m-%d'))
        for iline in datedata:
            args_lst.append((iline, args.url, args.thresval, args.outputdir, start_time))
    pool.starmap_async(combining_fn, args_lst)
    pool.close()
    pool.join()
