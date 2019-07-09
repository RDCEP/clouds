#
# + save function frequently using for analyssis relating timestamp
#
import os
import datetime
import numpy as np

def date2deltaday(year1, month1, day1,
                  year2, month2, day2,
                  modis_type=True
    ):
    """
    INPUTS
    """
    a = datetime.date(year1, month1, day1)
    b = datetime.date(year2, month2, day2)
    if modis_type:
      return int((b-a).days) + 1 # index of dates 
    else:
      return int((b-a).days) # just diff of days


def deltaday2date(cyear, timestamp, str_flag=False):
  """
  INPUT:
    cyear    : int
    timestamp: int, index of date
    str_flag : bool, if true, fill 0 to ten digit 

  OUTPUTS: 
    year : int 
    month: int
    day  : int
  """
  
  # intial date
  date = datetime.datetime(cyear, 1, 1)
  # get date-info
  date += datetime.timedelta(timestamp-1) # -1 since diff
  
  if str_flag:
    if date.month < 10:
       month = '0'+str(date.month)
    if date.day < 10:
       day = '0'+str(date.day)
    return str(date.year), month, day
  else:
    return date.year, date.month, date.day
