# _*_ coding : utf-8 _*_

import os
import sys
import numpy as np
import random
import argparse
import time


p = argparse.ArgumentParser()
p.add_argument(
  '--ndays',
  help='number of days generating for output datalist',
  type=int,
  default=1
)
p.add_argument(
  '--outdir',
  help='name of output data directory',
  type=str,
  default=os.getcwd()
)
p.add_argument(
  '--oname',
  help='name of output txt file',
  type=str,
  default='output'
)
args = p.parse_args()

def get_proptime(stime='2000-02-24', 
                 etime='2019-03-27', 
                 _format='%Y-%m-%d',
                 prop = 1.0
  ):
  """prop specifies how a proportion of the interval 
     to be taken after start.  Range is between 0-1(float)
     The returned time will be in the specified format.
  """
  sdate = time.mktime(time.strptime(stime, _format))
  edate = time.mktime(time.strptime(etime, _format))

  ptime = sdate + prop*(edate - sdate)

  return time.strftime(_format, time.localtime(ptime))

def get_missing_dates(*args):
  days_list = [
      "2000-08-06",
      "2000-08-07",
      "2000-08-08",
      "2000-08-09",
      "2000-08-10",
      "2000-08-11",
      "2000-08-12",
      "2000-08-13",
      "2000-08-14",
      "2000-08-15",
      "2000-08-16",
      "2000-08-17",
      "2002-10-17",
      "2016-02-20",
      "2016-02-21",
      "2016-02-22",
      "2016-02-23",
      "2016-02-24",
      "2016-02-25",
      "2016-02-26",
      "2016-02-27",
  ]
  return days_list

def gen_randomDate(ndays=1,stime='2000-02-24', etime='2019-03-27', _format='%Y-%m-%d', rand_seed=123456):

  filelist = []
  missing_date_list = get_missing_dates() # avoid missing data days
  random.seed(rand_seed)
  while len(filelist) < ndays:
    ctime = get_proptime(
                 stime = stime, etime = etime, _format= _format, prop=random.random()
    )
    if not ctime in missing_date_list:
      filelist.append(ctime)
    # original code
    #filelist += [get_proptime(
    #             stime = stime, etime = etime, _format= _format, prop=random.random()
    #)]
  filelist.sort()
  return filelist


def save_filelist(filelist=[], outdir='./', oname='output' ):
  with open(outdir+"/"+oname+".txt", 'w') as ofile:
    for iline in filelist:
        ofile.write(iline+'\n')
  print( " ### File Saved %s/%s.txt " % (outdir ,  oname) )
 

if __name__ == "__main__":
  # settings
  ndays = args.ndays
  outdir = args.outdir
  oname  = args.oname
  print("   Number of days to download: %d" % ndays )

  filelist = gen_randomDate(ndays, stime='2003-01-01')
  save_filelist(filelist, outdir=outdir, oname=oname)
  