import os
import sys
import argparse
import datetime
import urllib
import time
import csv
from subprocess import Popen, PIPE, run, call, check_output
from io import StringIO
from urllib.request import urlopen
import multiprocessing as mp
from bs4 import BeautifulSoup
import requests
import pandas as pd
import prg_gen_rndm_metadata as pgrm

#            'MYD02': 'https://ladsweb.modaps.eosdis.nasa.gov/archives/allData/61/MYD021KM',
#            'MYD35': 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD35_L2', 
#            'MYD06': 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD06_L2'}

def download_data(url, outputdir='.', appkey=None):
   os.makedirs(outputdir, exist_ok=True)
   start_time = datetime.datetime.now()
   try:
     BASHSCRIPT='./easy_execute2.bash'
     p = Popen([ 'bash',BASHSCRIPT, url,appkey,outputdir ], stdout=PIPE, stderr=PIPE)
   except Exception as e:
     print(e)

   result = p.communicate()

   print("## FINISH DOWNLOAD ! ## ",  flush=True)
   curr_time = datetime.datetime.now()
   print(pd.Timedelta(curr_time - start_time), flush=True)


if __name__ == "__main__":
  #url = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD021KM/2008/001/'
  #appkey = '210B0BEC-13E1-11EA-B059-AE780D77E571'
  #outputdir="/home/tkurihana/Research/data/MYD02"
  #download_data(url, outputdir=outputdir, appkey=appkey)

  dates_list = [
    '053','068','121','122','168','189','235','272','282','312','337'
  ]
  for date in dates_list:
    url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD35_L2/2008/{date}/"
    appkey = '210B0BEC-13E1-11EA-B059-AE780D77E571'
    outputdir="/home/tkurihana/Research/data/MYD35"
    download_data(url, outputdir=outputdir, appkey=appkey)
