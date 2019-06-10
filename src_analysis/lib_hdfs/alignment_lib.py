# _*_ coding: utf-8_*_
#
# library for data alignment with MOD35 Cloud Fraction Data
#

import numpy as np
from pyhdf.SD import SD, SDC

def decode_cloud_flag(sds_array, fillna=True):
    """ Assume sds_array = hdf.select('Cloud_Mask')
        File: Cloud_Mask_1.hdf which stores first important 6bits
        
        +Flags
        0: 00 = cloudy
        1: 01 = uncertain clear
        2: 10 = probably clear
        3: 11 = confident clear
    """
    def bits_stripping(bit_start,bit_count,value):
        bitmask=pow(2,bit_start+bit_count)-1
        return np.right_shift(np.bitwise_and(value,bitmask),bit_start)
    cm1_array = sds_array.get()
    nx, ny = cm1_array.shape
    carray = np.zeros((nx,ny))
    for ix in range(nx):
        for iy in range(ny):
            cloud_mask_flag = bits_stripping(1,2,cm1_array[ix,iy])
            carray[ix, iy] = cloud_mask_flag
    ncarray = carray.astype(np.float64)
    if fillna:
        nan_idx = np.where(cm1_array == sds_array.attributes()['_FillValue'])
        ncarray[nan_idx] = np.nan
    return ncarray


def _decode_cloud_flag(sds_array, fillna=True):
    """ Assume sds_array = hdf.select('Cloud_Mask') [6,nx,ny]
         File: Cloud_Mask_1.hdf which stores first important 6bits

         +Flags
         0: 00 = cloudy
         1: 01 = uncertain clear
         2: 10 = probably clear
         3: 11 = confident clear
    """
    def bits_stripping(bit_start,bit_count,value):
        bitmask=pow(2,bit_start+bit_count)-1
        return np.right_shift(np.bitwise_and(value,bitmask),bit_start)
    cm_array = sds_array.get()
    _, nx, ny = cm_array.shape
    carray = np.zeros((nx,ny))
    for ix in range(nx):
        for iy in range(ny):
            cloud_mask_flag = bits_stripping(1,2,cm_array[0,ix,iy])
            carray[ix, iy] = cloud_mask_flag
    ncarray = carray.astype(np.float64)
    if fillna:
        nan_idx = np.where(cm_array[0] == sds_array.attributes()['_FillValue'])
        ncarray[nan_idx] = np.nan
    return ncarray

def const_clouds_array(patches, clouds_mask, width=128, height=128, thres=0.2):
    """
    thres: range 0-1. ratio of clouds within the given patch
    dev_const_clouds_array in analysis_mode021KM/016
    """
    nx, ny = patches.shape[:2]
    patches_list = []
    xy_list = []
    for i in range(nx):
        for j in range(ny):
            if not np.isnan(patches[i,j]).any():
                if np.any(clouds_mask[i*width:(i+1)*width,j*height:(j+1)*height] == 0):
                    tmp = clouds_mask[i*width:(i+1)*width,j*height:(j+1)*height]
                    nclouds = len(np.argwhere(tmp == 0))
                    if nclouds/(width*height) > thres:
                        patches_list += [patches[i,j]]
                        xy_list += [(i,j)]
    return patches_list, xy_list
