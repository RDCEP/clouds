# _*_ coding: utf-8   _*_
# 
# library for hdf data load and retrive data in correct manner
#
import numpy as np
from pyhdf.SD import SD, SDC

def _get_hdf4(datadir, filename):
    hdf = SD(datadir+'/'+filename+'.hdf', SDC.READ)   # SD object 
    return hdf

def _retrive_data(ndvi_sds):
    """
    IN:
    ndvi_sds = cth
    
    TMP:
    ndvi = cth.get()
    
    RETURN:
    float_value (np.ndaeeay)
    
    * float_value equation*
    float value = scale_factor * (stored integer - add_offset)
    
    """
    ndvi = ndvi_sds.get()
    # locate no-data (fill value)
    nan_index=np.where(ndvi==ndvi_sds.attributes()['_FillValue'])  
    # locate offset
    addoffset_index = np.where(ndvi == ndvi_sds.attributes()['add_offset'] )
    #  tmp_float_value = stored_int - add_offset
    ndvi = ndvi - ndvi_sds.attributes()['add_offset']
    #  float_value = tmp_float_value * scale_factor
    ndvi=ndvi * ndvi_sds.attributes()['scale_factor']   # apply scaling (digital number to real data)
    #  fill-val to np.NAN
    ndvi[nan_index]=np.nan
    return ndvi
