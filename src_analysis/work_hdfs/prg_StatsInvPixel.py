# _*_ coding: utf-8 _*_ 

import os
import re
import sys
import glob
import argparse
import csv
import numpy as np
from pyhdf.SD import SD, SDC

# my libraries
hdf_libdir = '/Users/katykoeing/Desktop/clouds/src_analysis/lib_hdfs' # change here
sys.path.insert(1,os.path.join(sys.path[0],hdf_libdir))
from alignment_lib import _gen_patches
from alignment_lib import gen_mod35_img 

""" Flow of this src-code
  1. get mod02 file
  2. decode mod02 file
  3. detect number of invalid pixel; 32767 < x < _FillValue
  4. if #x > 0 go to step 5~, else next file
  5. gen mod02 patch
  6. decode mod35
  7. alignment mod02-mod35
  8. get patches including invalid pixels
  9. export file with filename(mod02), # of inv-pixels, # of inv-patches  
"""

def _proc_sds(sds_array,sdsmax=32767):
    #
    # trainig patch version
    #
    """
    IN: array = hdf_data.select(variable_name)
    OUT: int = flag inv
    """
    array = sds_array.get()
    array = array.astype(np.float64)
    
    #TODO: do not erase comment out since we need "invalid pixel"
    # invalid value
    #err_idx = np.where( (array > sdsmax)
    #    & (array < sds_array.attributes()['_FillValue']) )
    # nan process
    nan_idx = np.where( array == sds_array.attributes()['_FillValue'])
    fillvalue = sds_array.attributes()['_FillValue']
    if len(nan_idx) > 0:
        array[nan_idx] = np.nan
    else:
        pass
    #if len(err_idx) > 0:
    #    array[err_idx] = np.nan
    #else:
    #    pass
    
    # number of invalid pixel
    #flag_array = np.zeros((array.shape[0]))
    #for iband in range(array.shape[0]):
    #    tmp_array = array[iband]
    #    # invalid index
    #    invalid_idx = np.where( tmp_array > sdsmax )
    
    #    # high value nan + more than 32767
    #    high_val_array = tmp_array[invalid_idx]
    
    #    #  only larger than 32767 and not np.nan
    #    invalid_array = high_val_array[~np.isnan(high_val_array)]
        
    #    # get number of invalid pixel
    #    flag_array[iband] = len(invalid_array)
    return fillvalue , array

def gen_mod02_img(filename):
    hdf = SD(filename, SDC.READ)

    # refsb
    refsb = hdf.select("EV_500_Aggr1km_RefSB")
    fillvalue_refsb, refsb_flag_array = _proc_sds(refsb)
    refsb_bands = [3,4,5,6,7]
    
    for idx, refsb_band in enumerate(refsb_bands):
        if refsb_band == 6:
            b6_array = refsb_flag_array[idx]
        elif refsb_band == 7:
            b7_array = refsb_flag_array[idx]
    
    # emissive
    ev = hdf.select("EV_1KM_Emissive")
    fillvalue_ev, ev_flag_array = _proc_sds(ev)
    ev_bands = [20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36]
    
    for idx, ev_band in enumerate(ev_bands):
        if ev_band == 20:
            b20_array = ev_flag_array[idx]
        elif ev_band == 28:
            b28_array = ev_flag_array[idx]
        elif ev_band == 29:
            b29_array = ev_flag_array[idx]
        elif ev_band == 31:
            b31_array = ev_flag_array[idx]
    
    # fill value list
    fillvalue_list = [
      fillvalue_refsb, 
      fillvalue_refsb, 
      fillvalue_ev, 
      fillvalue_ev, 
      fillvalue_ev, 
      fillvalue_ev, 
    ]

    # gen mod02_img without scaling
    # size adjust for 
    nx, ny = b6_array.shape
    d_list = [
        b6_array.reshape(nx,ny,1),
        b7_array.reshape(nx,ny,1),
        b20_array.reshape(nx,ny,1),
        b28_array.reshape(nx,ny,1),
        b29_array.reshape(nx,ny,1),
        b31_array.reshape(nx,ny,1),
    ]
    mod02_img = np.concatenate(d_list, axis=2)
    # check 
    return fillvalue_list , mod02_img


def get_filepath(filepath, datadir, prefix='MOD35_L2.A'):
    """filepath for another modis data corresponding given filepath
    """
    bname = os.path.basename(filepath) # ex. 2017213.2355
    dateinfo = re.findall('[0-9]{7}.[0-9]{4}' , bname)
    cdate = dateinfo[0].split('.')[0].rstrip("['")
    ctime = dateinfo[0].split('.')[1].lstrip("]'")
    date = cdate+'.'+ctime
    #date = bname[10:22]
    return glob.glob(datadir+'/'+prefix+date+'*.hdf')[0]


def check_invalid_clouds_array(patches, clouds_mask, fillvalue_list,
                               width=128, height=128, thres=0.3,sdsmax=32767):
    """
    thres: range 0-1. ratio of clouds within the given patch
    dev_const_clouds_array in analysis_mode021KM/016
    """
    nx, ny = patches.shape[:2]
    patches_list = []   # just insert 1
    inv_pixel_list = [] # number of invalid pixel
    for i in range(nx):
        for j in range(ny):
            #if not np.isnan(patches[i, j]).any():
                if np.any(clouds_mask[i*width:(i+1)*width, j*height:(j+1)*height] == 0):
                    tmp = clouds_mask[i*width:(i+1)*width, j*height:(j+1)*height]
                    nclouds = len(np.argwhere(tmp == 0))
                    if nclouds/(width*height) > thres:
                      # valid patches. Search number of invalid pixel
                      patches_list += [1]
                      # search invalid pixel
                      # NOTE here number of pixel sums up each layer
                      n_inv_pixel = 0
                      for iband in range(6):
                        tmp_array = patches[i, j, :, :, iband]
                        tmp_fillvalue = fillvalue_list[iband]                      
                        err_idx = np.where((tmp_array > sdsmax) & (tmp_array < tmp_fillvalue))
                        n_inv_pixel += len(err_idx[0]) # should state 0
                      # sum up!
                      inv_pixel_list += [n_inv_pixel]
    return patches_list, inv_pixel_list

def check_invalid_clouds2(output_file, file, patches, clouds_mask, fillvalue_list, width=128, height=128, thres=0.3, sdsmax=32767):
    """
    thres: range 0-1. ratio of clouds within the given patch
    dev_const_clouds_array in analysis_mode021KM/016
    """
    with open(output_file, 'a') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        nx, ny = patches.shape[:2]
        patch_counter = 0 
        patches_list = []   # just insert 1
        inv_pixel_list = [] # number of invalid pixel
        for i in range(nx):
            for j in range(ny):
                #if not np.isnan(patches[i, j]).any():
                if np.any(clouds_mask[i*width:(i+1)*width, j*height:(j+1)*height] == 0):
                    tmp = clouds_mask[i*width:(i+1)*width, j*height:(j+1)*height]
                    nclouds = len(np.argwhere(tmp == 0))
                    if nclouds/(width*height) > thres:
                      # valid patches. Search number of invalid pixel
                      patches_list += [1]
                      # search invalid pixel
                      # NOTE here number of pixel sums up each layer
                      n_inv_pixel = 0
                      for iband in range(6):
                        tmp_array = patches[i, j, :, :, iband]
                        tmp_fillvalue = fillvalue_list[iband]
                        #err2 = tmp_array[(tmp_array > sdsmax) & (tmp_array < tmp_fillvalue)]
                        err_idx = np.where((tmp_array > sdsmax) & (tmp_array < tmp_fillvalue))
                        n_inv_pixel += len(err_idx[0]) # should state 0
                    inv_pixel_list.append(n_inv_pixel)
                    outputwriter.writerow([file, patch_counter, n_inv_pixel])
                    patch_counter += 1
    csvfile.close()
    return patches_list, inv_pixel_list



def save_file(filename, outputdir, outputname, inv_pixel_list, patches_list):
  # save as file
  date = gen_date(filename)
  np.savez(outputdir+"/"+outputname+'_'+date, filename=np.asarray([filename]),
           pixel_list=np.asarray(inv_pixel_list), patch_list=np.asarray(patches_list))
  # show to check
  n_inv_pixel = np.sum(inv_pixel_list)
  n_inv_patch = np.sum([patches_list[idx] for idx, i in enumerate(inv_pixel_list) if i > 0])
  print(" ### SAVE FILE %s ==> # inv. pixel: %d | # inv patch: %d ###" 
          % (filename, n_inv_pixel , n_inv_patch ))


def gen_date(filepath):
    bname = os.path.basename(filepath) # ex. 2017213.2355
    dateinfo = re.findall('[0-9]{7}.[0-9]{4}' , bname)
    cdate = dateinfo[0].split('.')[0].rstrip("['")
    ctime = dateinfo[0].split('.')[1].lstrip("]'")
    return cdate+'.'+ctime


def run_main(inputdir = './', mod35_datadir ='./', outputdir= './', outputname='output'):
  # get filelist
  filelist = glob.glob(inputdir+'/*.hdf')
  # make save directory
  os.makedirs(outputdir, exist_ok=True)
  for filename in filelist:
    # get number of mod02
    fillvalue_list, mod02_img = gen_mod02_img(filename)
    # get corresponding filepath
    #print(m2_file, args.mod35_datadir)
    m35_file = get_filepath(filename, mod35_datadir)
    # hdf mod35
    if not os.path.exists(m35_file):
      print(os.path.basename(m35_file))
      raise NameError(" ! Program Terminated: File does not exist!" )
    hdf_m35 = SD(m35_file, SDC.READ)
    # get clouds mask array
    clouds_mask_img = gen_mod35_img(hdf_m35)    
    # patches
    mod02_patches = _gen_patches(mod02_img, normalization=False)
    # examine number of invalid patches
    # 1. cloud patches with at least ${thres} percent of cloud flag within patch
    # 2. within these patches, count number of patches with invalid pixel
    patches_list, inv_pixel_list = check_invalid_clouds_array(
          mod02_patches, clouds_mask_img, fillvalue_list, thres=0.3)
    # save file
    save_file(filename,outputdir,outputname,inv_pixel_list, patches_list)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument(
    "--mod02_datadir",
    type=str,
  )
  p.add_argument(
    "--mod35_datadir",
    type=str,
  )
  p.add_argument(
    "--outputdir",
    type=str,
  )
  p.add_argument(
    "--outputname",
    type=str,
  )
  args = p.parse_args()

  # main
  run_main(inputdir=args.mod02_datadir, mod35_datadir=args.mod35_datadir,
           outputdir=args.outputdir, outputname=args.outputname)
