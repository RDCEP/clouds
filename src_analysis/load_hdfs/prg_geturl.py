# _*_  coding: utf-8  _*_
import os
import argparse
import datetime
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup

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
      if keyward in i  and ".hdf" in i :
        hrefs+=[i]
        print(i)
  return hrefs

def bool_fsize(url, thresval=100, mode='above'):
    """
    IN : url
         thresval ; thre value of file size (MB)
         mode ; return file requirement
                [Options]
                  above(above thresval)
                  below(below thresval) 
    """
    coef = 1/1000/1000
    site = urlopen(url)
    fsize = site.length*coef
    if fsize >= thresval:
        flag = True
    else:
        flag = False
    return flag

def download_data(url, outputdir='.'):
    filename = os.path.basename(url)
    urllib.request.urlretrieve(url, os.path.join(outputdir, filename))
    print("## FINISH DOWNLOAD ! ## %s" % filename)

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
  args = p.parse_args() 
  outputdir = args.outputdir
  os.makedirs(outputdir, exist_ok=True)


  # load date-meta text file
  date_list = []
  with open(args.datedata, 'r') as ifile:
     for iline in ifile.readlines():
        idate = iline.split('\n')[0]
        #date_list.append([ idate[:4], idate[5:7], idate[8:] ])
        date_list.append([idate[:4], 
                         delta_day(year=idate[:4], month=idate[5:7], day=idate[8:])
                         ])
  for idate in date_list:
    # URL
    url = args.url+'/'+idate[0]+'/'+idate[1]+'/'
  
    # href_lists
    href_list = get_href_lists(url, keyward=args.keyward)

    # select url over 100M
    for ihref in href_list:
      https = url+os.path.basename(ihref)
      bfsize = bool_fsize(https,thresval=args.thresval)
      if bfsize:
        download_data(https, outputdir=outputdir) 
   
