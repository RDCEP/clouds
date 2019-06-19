# _*_  coding: utf-8  _*_
import os
import argparse
import datetime
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import multiprocessing as mp
import datetime
import pandas as pd


def get_href_lists(url, keyward="MOD021KM.A"):
  """
  IN  : url; url
  OUT : hrefs; href with keyward
  """
  https = urlopen(url)
  bsobj = BeautifulSoup(https, 'html.parser')
  # get href
  _hrefs = []
  for i in bsobj.findAll("a"):
    _hrefs += [ i.get("href") ]
  # select href
  hrefs = []
  for i in _hrefs:
    if isinstance(i, str):
      if keyward in i  and ".hdf" in i:
        hrefs+=[i]
        print(i)
  return hrefs


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


def bool_fsize(url, thresval=100, mode='above'):
  """
  IN : url
       thresval ; thre value of file size (MB)
       mode ; return file requirement
              [Options]
                above(above thresval)
                below(below thresval)
  Returns: boolean
  """
  coef = 1/1000/1000
  site = urlopen(url)
  fsize = site.length*coef
  return fsize >= thresval


def bool_fsize2(url, thresval=100):
  '''
  Checks that webpage is larger than a certain threshold 

  Inputs:
    url: url for website
    thresval: threshold value

  Outputs: boolean
  '''
  coef = 1/1000/1000
  r = requests.get(url)
  fsize = len(r.content)
  return fsize >= thresval


def download_data(url, start_time, outputdir='.'):
  filename = os.path.basename(url)
  urllib.request.urlretrieve(url, os.path.join(outputdir, filename))
  print("## FINISH DOWNLOAD ! ## %s" % filename)
  curr_time = datetime.datetime.now()
  print(pd.Timedelta(curr_time - start_time))


# def download_data2(url, start_time, outputdir='.'):
#   '''
#   Downloads images from specified website to specified directory

#   Inputs:
#     url: a string presenting a url
#     start_time: datetime object representing when entire process began
#     outputdir: string of directory name

#   Outputs: None (saves images)
#   '''
#   filename = os.path.basename(url)
#   r = requests.get(url)
#   i = Image.open(BytesIO(r.content))
#   i.save(filename)
#   print("## FINISH DOWNLOAD ! ## %s" % filename)
#   curr_time = datetime.datetime.now()
#   print(pd.Timedelta(curr_time - start_time))


def delta_day(year='0000',month='00', day='00'):
  """
  Return # days of input date by the scale of 365
  e.g. 2000/1/10 = 10, 2000/12/31 = 365
  """
  _year  = int(year)
  _month = int(month)
  _day   = int(day)
  delta = datetime.date(_year, _month, _day) - datetime.date(_year, 1,1)
  dday = delta.days+1
  if dday < 10:
    delta_day = "00"+str(dday)
  elif dday >= 10 and dday < 100:
    delta_day = "0"+str(dday)
  elif dday >= 100 and dday < 1000:
    delta_day = str(dday)
  return delta_day


def combining_fn(href_list, ihref, url, thresval, outputdir, start_time):
  '''
  Download requests HDF files into specified directory

  Inputs:
    iline
    base_url
    keyward
    thresval
    outputdir

  Outputs: saved .out file with list HDF files downlading
  '''
  https = url + os.path.basename(ihref)
  bfsize = bool_fsize2(https, thresval)
  if bfsize:
    response = requests.get(https)
    if response.status_code == 200:
      print('about to download')
      download_data(https, start_time, outputdir)
    else:
      href_list.append(ihref)



if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument(
    '--url',
    help=' base url without datenumber and year',
    type=str
  )
  p.add_argument(
    '--outputdir',
    type=str,
    default=os.getcwd()
  )
  p.add_argument(
    '--datedata',
    type=str
  )
  p.add_argument(
    '--keyward',
    type=str,
    default='MOD021KM.A'
  )
  p.add_argument(
    '--thresval',
    type=int,
    default=100
  )
  p.add_argument(
  '--processors',
  type=int,
  default=7
  )
  args = p.parse_args()
  os.makedirs(args.outputdir, exist_ok=True)
  start_time = datetime.datetime.now()
  print(start_time)
  print(args.processors)
  # Initializes pooling proess for parallelization
  pool = mp.Pool(processes=args.processors)  
  # Loads date metadata
  with open(args.datedata, 'r') as ifile:
    for iline in ifile.readlines():
      date = iline.split('\n')[0]
      year = date[:4]
      days = delta_day(year=date[:4], month=date[5:7], day=date[8:])
      url = args.url+'/'+year+'/'+days+'/'
      # Check if valid url
      response = requests.get(url)
      if response.status_code == 200:
        # href_lists
        href_list = get_href_lists2(url, args.keyward)
        for ihref in href_list[:20]:
          pool.apply(combining_fn, args=(href_list, ihref, url, args.thresval, args.outputdir, start_time))
      else:
        # Adds dates w/o data available to running list of dates
        with open('no-data-dates.txt', 'a') as f:
          f.write(str(date) + "\n")
  pool.close()
  pool.join()
  ifile.close()
