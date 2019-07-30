# _*_ coding: utf-8_*_
#
# library for data alignment with MOD35 Cloud Fraction Data
# 
# + Description
#   07/30/2019: Update patch-wise mean method from prg_sklearn_randomforest.oy
#
import os
import gc
import re
import sys
import glob
import copy
import numpy as np
import scipy as sc
from pyhdf.SD import SD, SDC

libdir = '/home/tkurihana/scratch-midway2/clouds/src_analysis/lib_hdfs'
sys.path.insert(1,os.path.join(sys.path[0],libdir))
from analysis_lib import mod06_proc_sds
from analysis_lib import _gen_patches


# 2d array version [ix, iy]
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

# 3d array version [6,ix,iy]
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

def const_clouds_array(patches, clouds_mask, width=128, height=128, thres=0.3):
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

# new function for into_mod_normed_record
def translate_const_clouds_array(patch, clouds_mask, width=128, height=128, thres=0.3, coord=(0,0)):
    """
    1 patch - 1 cloud-flags/patch.
    thres: range 0-1. ratio of clouds within the given patch
    function to return one clouds_patch and clouds-flag based on corrdinate (i,j)
    
    OUT:
      clouds_patch: np.array(128,128,nband)
      clouds_flag : Boolean{True; valid cloud patch, False; o.w.}  
    """
    # coordinates
    # this (i,j) is i in [0,2030,width] & [0,1354,height]
    i,j = coord

    # prep flag
    clouds_flag = False

    # main process
    if np.any(clouds_mask[i:i+width,j:j+height] == 0):
        tmp = clouds_mask[i:i+width,j:j+height]
        nclouds = len(np.argwhere(tmp == 0))
        if nclouds/(width*height) > thres:
          # valid patch
          clouds_flag = True
          # TODO: if not necessary below 2 lines, erase
          #if patch is None:
          #  print("Nonetype")
          return patch, clouds_flag    
        else:
          # return patch and false
          return patch, clouds_flag    
    else:
      # return patch and false
      return patch, clouds_flag    
       

# below, from prg_augment_2 

def mod02_proc_sds(sds_array):
    """
    IN: array = hdf_data.select(variable_name)
    """
    array = sds_array.get()
    array = array.astype(np.float64)
    
    # nan process
    nan_idx = np.where( array == sds_array.attributes()['_FillValue'])
    if len(nan_idx) > 0:
        array[nan_idx] = np.nan
    else:
        pass
    # invalid value process
    # TODO: future change 32767 to other value
    invalid_idx = np.where( array > 32767 )
    if len(nan_idx) > 0:
        array[invalid_idx] = np.nan
    else:
        pass
    
    # radiacne offset
    offset = sds_array.attributes()['radiance_offsets']
    offset_array = np.zeros(array.shape) # new matrix
    offset_ones  = np.ones(array.shape)  # 1 Matrix 
    for iband in range(len(offset)):
        offset_array[iband, :,:] = array[iband, :,:] - offset[iband]*offset_ones[iband,:,:]
    
    # radiance scale
    scales = sds_array.attributes()['radiance_scales']
    scales_array = np.zeros(array.shape) # new matrix
    for iband in range(len(offset)):
        scales_array[iband,:,:] = scales[iband]*offset_array[iband,:,:]
    
    #del nan_idx, offset, scales, offset_ones
    return scales_array

def gen_mod02_img(hdf):
    # band RefSB
    refsb_sds = hdf.select("EV_500_Aggr1km_RefSB")
    refsb_array = mod02_proc_sds(refsb_sds)
    refsb_bands = [3,4,5,6,7]
    
    # band Emissive
    ev_sds = hdf.select("EV_1KM_Emissive")
    ev_array = mod02_proc_sds(ev_sds)
    ev_bands = [20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36]
    
    # band selection
    # + RefSB
    for idx, ref_band in enumerate(refsb_bands):
        if ref_band == 6:
            b6_array = refsb_array[idx]
        elif ref_band == 7:
            b7_array = refsb_array[idx]

    # + Emissive
    for idx, ev_band in enumerate(ev_bands):
        if ev_band == 20:
            b20_array = ev_array[idx]
        elif ev_band == 28:
            b28_array = ev_array[idx]
        elif ev_band == 29:
            b29_array = ev_array[idx]
        elif ev_band == 31:
            b31_array = ev_array[idx] 
    
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
    return mod02_img

def gen_mod06_img(hdf):
    cot_sds = hdf.select("Cloud_Optical_Thickness")
    cot_array = mod06_proc_sds(cot_sds,variable="Cloud_Optical_Thickness")
    
    cwp_sds = hdf.select("Cloud_Water_Path")
    cwp_array = mod06_proc_sds(cwp_sds)
    
    cpi_sds = hdf.select("Cloud_Phase_Infrared_1km")
    cpi_array = mod06_proc_sds(cpi_sds)   

    ctp_sds = hdf.select("cloud_top_pressure_1km")
    ctp_array = mod06_proc_sds(ctp_sds) 
    
    #print(cpi_array.shape)
    nx, ny = cot_array.shape
    d_list = [
        cot_array.reshape(nx,ny,1),
        cwp_array.reshape(nx,ny,1),
        cpi_array.reshape(nx,ny,1),
        ctp_array.reshape(nx,ny,1),
    ]
    mod06_img = np.concatenate(d_list, axis=2)
    return mod06_img

def gen_mod35_img(hdf):
    cm_sds = hdf.select('Cloud_Mask')
    clouds_mask_array = _decode_cloud_flag(cm_sds)
    return clouds_mask_array

def gen_mod35_img_single(
      hdf_datadir='hdf data directory',
      date='2015001' 
    ):
    mod35_file = glob.glob(hdf_datadir+'/MOD35_L2.A'+date+'.mosaic.061*.Cloud_Mask_1.hdf')[0]
    hdf    = SD(mod35_file, SDC.READ)
    cm_sds = hdf.select('Cloud_Mask')
    clouds_mask_array = decode_cloud_flag(cm_sds)
    return clouds_mask_array


def get_filepath(filepath, datadir, prefix=''):
    """filepath for another modis data corresponding given filepath
    """
    #FIXME
    #date = os.path.basename(filepath)[10:22] # ex. 2017213.2355
    bname = os.path.basename(filepath) # ex. 2017213.2355
    dateinfo = re.findall('[0-9]{7}.[0-9]{4}' , bname)
    date     = dateinfo[0].rstrip("['").lstrip("']")
    filelist = glob.glob(datadir+'/'+prefix+date+'*.hdf')
    try:
      if len(filelist) > 0:
        # in the case file-exist
        return filelist[0]
    except:
      efile = datadir+'/'+prefix+date+'*.hdf'
      print(" Program will be forcibly terminated", flush=True)
      raise NameError(" ### File Not Found: No such file or directory "+str(efile)+" ### ")
    #return glob.glob(datadir+'/'+prefix+date+'*.hdf')[0]


def compute_augment(encoder, m2_file, mod35_datadir='./', thres=0.3, height=128, width=128, bands=6): 
    """
    """
    # hdf mod02
    hdf_m2 = SD(m2_file, SDC.READ)

    # get corresponding filepath
    #print(m2_file, args.mod35_datadir)
    m35_file = get_filepath(m2_file, mod35_datadir)

    # hdf mod35
    if not os.path.exists(m35_file):
        print(os.path.basename(m35_file))
    hdf_m35 = SD(m35_file, SDC.READ)
  
    # get image process
    mod02_img = gen_mod02_img(hdf_m2)
    clouds_mask_img = gen_mod35_img(hdf_m35)

    # patches
    mod02_patches = _gen_patches(mod02_img, normalization=False)
  
    # cloud patches with at least ${thres} percent of cloud flag within patch
    clouds_patches_list, clouds_xy_list = const_clouds_array(mod02_patches, clouds_mask_img, thres=thres)

    if len(clouds_patches_list):
      """
      Case when mod02 patch is valid/no-nan
      """
      encs_list = []
      for i in clouds_patches_list:
        encs =  encoder.predict(i.reshape(1,height,width,bands))
        encs_list += [encs.mean(axis=(1,2))]
      features = np.concatenate(encs_list, axis=0)
  
      print(features.shape)    
      
      return  features, clouds_xy_list, m2_file
    else:
      features = []
      return  features, clouds_xy_list, m2_file


def gen_mod02_img_sigle(
      hdf_datadir='hdf data directory',
      date='2015001' 
    ):

    #ad-hoc bands assumed 6,7,20,28,29 and 31 
    _refsb_list = [
      hdf_datadir+'/MOD021KM.A'+date+'.mosaic.061.*.EV_500_Aggr1km_RefSB_4.hdf',
      hdf_datadir+'/MOD021KM.A'+date+'.mosaic.061.*.EV_500_Aggr1km_RefSB_5.hdf',
    ]
    refsb_list = [ glob.glob(i)[0] for i in _refsb_list]

    _ev_list = [
      hdf_datadir+'/MOD021KM.A'+date+'.mosaic.061.*.EV_1KM_Emissive_1.hdf',
      hdf_datadir+'/MOD021KM.A'+date+'.mosaic.061.*.EV_1KM_Emissive_8.hdf',
      hdf_datadir+'/MOD021KM.A'+date+'.mosaic.061.*.EV_1KM_Emissive_9.hdf',
      hdf_datadir+'/MOD021KM.A'+date+'.mosaic.061.*.EV_1KM_Emissive_11.hdf'
    ]
    ev_list = [ glob.glob(i)[0] for i in _ev_list]

    # band RefSB
    #refsb_sds = hdf.select("EV_500_Aggr1km_RefSB")
    refsb_array = []
    for ifile in refsb_list:
      refsb_hdf = SD(ifile, SDC.READ)
      refsb_sds = refsb_hdf.select("EV_500_Aggr1km_RefSB")
      refsb_array.append(mod02_proc_sds_single(refsb_sds))
    refsb_bands = [6,7]
    
    # band Emissive
    ev_array = []
    for ifile in ev_list:
      ev_hdf = SD(ifile, SDC.READ)
      ev_sds = ev_hdf.select("EV_1KM_Emissive")
      ev_array.append(mod02_proc_sds_single(ev_sds))
    ev_bands = [20,28,29,31]
    
    # band selection
    # + RefSB
    for idx, ref_band in enumerate(refsb_bands):
        if ref_band == 6:
            b6_array = refsb_array[idx][0]
        elif ref_band == 7:
            b7_array = refsb_array[idx][0]

    # + Emissive
    for idx, ev_band in enumerate(ev_bands):
        if ev_band == 20:
            b20_array = ev_array[idx][0]
        elif ev_band == 28:
            b28_array = ev_array[idx][0]
        elif ev_band == 29:
            b29_array = ev_array[idx][0]
        elif ev_band == 31:
            b31_array = ev_array[idx][0]
    
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
    return mod02_img


def mod02_proc_sds_single(sds_array):
    """
    IN: array = hdf_data.select(variable_name)
    """
    array = sds_array.get()
    array = array.astype(np.float64)
    
    # check bandinfo
    _bands = sds_array.attributes()['band_names']
    print("Process bands", _bands)
    bands = _bands.split(",")
    
    # nan process
    nan_idx = np.where( array == sds_array.attributes()['_FillValue'])
    if len(nan_idx) > 0:
        array[nan_idx] = np.nan
    else:
        pass
    # invalid value process
    # TODO: future change 32767 to other value
    invalid_idx = np.where( array > 32767 )
    if len(nan_idx) > 0:
        array[invalid_idx] = np.nan
    else:
        pass
    
    # radiacne offset
    offset = sds_array.attributes()['radiance_offsets']
    offset_array = np.zeros(array.shape) # new matrix
    offset_ones  = np.ones(array.shape)  # 1 Matrix 
    offset_array[:,:] = array[:,:] - offset*offset_ones[:,:]
    
    # radiance scale
    scales = sds_array.attributes()['radiance_scales']
    scales_array = np.zeros(array.shape) # new matrix
    scales_array[:,:] = scales*offset_array[:,:]
    return scales_array, bands

def _gen_patches(img, stride=128, size=128, 
                 normalization=False, flag_nan=True, isNoBackground=False,
                 verbose=0
    ):
    """ IF user get patches WITHOUT Normalization
         normalization = False
         Otherwise, vals in patch are normalized
        IF user want to get patches WITHOUT NAN value
         flag_nan=True
         Then, nanvalue will be excluded
    """
    # generate swath again
    swath = img   
    # Fix boolean option now
    if flag_nan:
      swath_mean = np.nanmean(swath, axis=(0,1))
      swath_std = np.nanstd(swath, axis=(0,1))
    else :
      swath_mean = swath.mean(axis=(0,1))
      swath_std = swath.std(axis=(0,1))
    # modify small std value 
    ill_stds = np.where(swath_std < 1.0e-20)[0]
    if len(ill_stds) > 0 :
        if verbose == 1:
          print("!====== Ill shape ======!")
          print(np.asarray(ill_stds).shape)
          print(ill_stds)  # coresponding to number of band
        for idx in ill_stds:
          swath_std[idx] += 1.0e-20
    patches = []

    stride = stride
    patch_size = size

    patches = []
    for i in range(0, swath.shape[0], stride):
        row = []
        for j in range(0, swath.shape[1], stride):
            if i + patch_size <= swath.shape[0] and j + patch_size <= swath.shape[1]:
                #p = swath[i:i + patch_size, j:j + patch_size].astype(float)
                if isNoBackground:
                  tmp_p = swath[i:i + patch_size, j:j + patch_size].astype(float)
                  # select only positice patch
                  if not np.all(tmp_p <= 1.0e-5):
                    p = tmp_p
                    if normalization:
                      p -= swath_mean
                      p /= swath_std
                    row.append(p)
                else:
                  p = swath[i:i + patch_size, j:j + patch_size].astype(float)
                  if normalization:
                    p -= swath_mean
                    p /= swath_std
                  row.append(p)
            
                #row.append(p)
        if row:
            patches.append(row)
    # original retuern        
    #return np.stack(patches)
    # Avoid np.stack ValueError if patches = []
    if patches:
      return np.stack(patches)
    else:
      return patches

# Add 2019/07/12
def prep_process_data(npz_dir='./', npzfname='clouds_patches_XXX.npz', 
                      mod06_datadir='./', prefix='MOD06_L2.A'
  ):

  # encs_mean & clouds_xy_list
  data_array = np.load(npz_dir+'/'+npzfname)
  encs_mean = data_array['encs_mean']
  clouds_xy = data_array['clouds_xy']

  # get mo06 file
  m6_file = get_filepath(npzfname, mod06_datadir, prefix=prefix)

  # mod06 file
  m6_hdf = SD(m6_file, SDC.READ)

  # gen train labels
  mod06_labels = gen_mod06_labels(m6_hdf, clouds_xy)
  gc.collect()

  return encs_mean, mod06_labels


def gen_mod06_labels(m6_hdf, clouds_xy, nparams=4):
  """
    INPUT: Opend MOD06 hdf.
  """
  #TODO: Feature, tie nparams and below read information
  # should be more flexible

  # resolution should be 1km
  cot_sds = m6_hdf.select("Cloud_Optical_Thickness")
  cer_sds = m6_hdf.select("Cloud_Effective_Radius")
  cwp_sds = m6_hdf.select("Cloud_Water_Path")
  ctp_sds = m6_hdf.select("cloud_top_pressure_1km")
  cpi_sds = m6_hdf.select("Cloud_Phase_Infrared_1km")

  # decode after scaling & offset & nan process
  cot_array = mod06_proc_sds(cot_sds, 'Cloud_Optical_Thickness')
  cer_array = mod06_proc_sds(cer_sds, 'Cloud_Effective_Radius')
  cwp_array = mod06_proc_sds(cwp_sds, 'Cloud_Water_Path')
  ctp_array = mod06_proc_sds(ctp_sds, 'cloud_top_pressure_1km')
  cpi_array = mod06_proc_sds(cpi_sds, 'Cloud_Phase_Infrared_1km')

  # integrate as one array 
  nx, ny = cot_array.shape
  d_list = [
    cot_array.reshape(nx,ny,1),
    cer_array.reshape(nx,ny,1),
    cwp_array.reshape(nx,ny,1),
    ctp_array.reshape(nx,ny,1),
    cpi_array.reshape(nx,ny,1),
  ]

  # concatenate
  mod06_img = np.concatenate(d_list, axis=2)

  # patchenaization
  mod06_patches = _gen_patches(mod06_img, normalization=False, flag_nan=True)
  mod06_patches_mean = np.nanmean(mod06_patches, axis=(2,3))
  prep_phase = copy.deepcopy(mod06_patches[:,:,:,:,-1])
  _x, _y = mod06_patches.shape[:2]
  phase_mode = np.zeros((_x,_y))
  for i in range(_x):
    for j in range(_y):
      phase, _= sc.stats.mode(prep_phase[i,j].ravel(), nan_policy='omit')
      phase_mode[i,j] = phase
  cpi_modes = np.zeros((_x,_y)).astype(np.float64)
  cpi_modes[:,:] = np.nan
  for idx, (x,y) in enumerate(clouds_xy):
    cpi_modes[x,y] = phase_mode[x,y]
  # finally join cpi mode to mod06_patches_mean
  mod06_patches_mean[:,:,-1] = cpi_modes
  
  # Process mod06 patches where align with clouds_xy information
  #index_list = []
  mod06_patches_valid = []
  for idx , i in enumerate(clouds_xy):
    ii,jj = i
    if not np.isnan(mod06_patches_mean[ii,jj]).any():
      mod06_patches_valid.append(mod06_patches_mean[ii,jj])
      #index_list.append(idx)


  #Finally get labels == patch-wise mean physics parameters
  #encs_labels = np.asarray(clist)
  encs_labels = np.asarray(mod06_patches_valid)

  return encs_labels

