# my function
import os
import re
import cv2
import glob
import numpy as np
from pyhdf.SD import SD, SDC

try:
  from numba import jit, generated_jit
except ImportError as  e:
  print('check if your virtual env has Numba', e)
  
########################################################
# Python module
########################################################

def gen_mod35_img(hdf):
    cm_sds = hdf.select('Cloud_Mask')
    clouds_mask_array = _decode_cloud_flag(cm_sds)
    return clouds_mask_array

def gen_mod35_ocean_img(hdf):
    cm_sds = hdf.select('Cloud_Mask')
    ocean_mask_array = _decode_ocean_flag(cm_sds)
    return ocean_mask_array

def gen_mod06_cloud_img(hdf,byte_index):
    cm_sds = hdf.select('Cloud_Mask_1km')
    clouds_mask_array = decode_cloud_flag(cm_sds,byte_index=byte_index)
    return clouds_mask_array

def gen_mod06_ocean_img(hdf,byte_index):
    cm_sds = hdf.select('Cloud_Mask_1km')
    ocean_mask_array = decode_ocean_flag(cm_sds,byte_index=byte_index)
    return ocean_mask_array

def get_filepath(filepath, datadir, prefix='',ext='.hdf'):
    """filepath for another modis data corresponding given filepath
    """
    bname = os.path.basename(filepath) # ex. 2017213.2355
    dateinfo = re.findall('[0-9]{7}.[0-9]{4}' , bname)
    date     = dateinfo[0].rstrip("['").lstrip("']")
    filelist = glob.glob(datadir+'/'+prefix+date+f'*{ext}')
    print(filelist)
    try:
        if len(filelist) > 0:
            # in the case file-exist
            return filelist[0]
    except:
        efile = datadir+'/'+prefix+date+'*.hdf'
        print(" Program will be forcibly terminated", flush=True)
        raise NameError(" ### File Not Found: No such file or directory "+str(efile)+" ### ")

# cloud mask decoder for MOD35 products
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

# cloud mask decoder for MOD06 products
@jit(cache=True)
def bits_stripping(bit_start,bit_count,value):
    bitmask=pow(2,bit_start+bit_count)-1
    return np.right_shift(np.bitwise_and(value,bitmask),bit_start)

def decode_cloud_flag(sds_array, fillna=True,byte_index=0):
    """ Assume sds_array = hdf.select('Cloud_Mask_1km') in MOD06 products
        Incoming sds array, a shape of (2030, 1354, 2) which stores first two important bytes
        first index byte #0 is exact copy as first byte of Cloud Mask product 
        
        +Flags
        0: 00 = cloudy
        1: 01 = uncertain clear
        2: 10 = probably clear
        3: 11 = confident clear
    """
    cm1_array = sds_array.get()
    cm1_array = np.squeeze(cm1_array[:,:,byte_index])
    ncarray = np.vstack(list(map(lambda x: bits_stripping(1,2,x), cm1_array))).astype(np.float64)
    if fillna:
        nan_idx = np.where(cm1_array == sds_array.attributes()['_FillValue'])
        ncarray[nan_idx] = np.nan
    return ncarray


def _decode_ocean_flag(sds_array, fillna=True):
    """ # 3d array version [6,ix,iy]
        Assume sds_array = hdf.select('Cloud_Mask') [6,nx,ny]
         File: Cloud_Mask_1.hdf which stores first important 6bits

     +Flags
     0: 00 = water
     1: 01 = coastal
     2: 10 = desert
     3: 11 = land
    """
    def bits_stripping(bit_start,bit_count,value):
        bitmask=pow(2,bit_start+bit_count)-1
        return np.right_shift(np.bitwise_and(value,bitmask),bit_start)
    cm_array = sds_array.get()
    _, nx, ny = cm_array.shape
    carray = np.zeros((nx,ny))
    for ix in range(nx):
        for iy in range(ny):
            # Shown below is the first byte only, of the six byte Cloud Mask
            # cm_array[index,ix,iy] : index should be 0 but get 6-7 bits of first  byte 
            cloud_mask_flag = bits_stripping(6,7,cm_array[0,ix,iy])
            carray[ix, iy] = cloud_mask_flag
    ncarray = carray.astype(np.float64)
    if fillna:
        nan_idx = np.where(cm_array[0] == sds_array.attributes()['_FillValue'])
        ncarray[nan_idx] = np.nan
    return ncarray

def decode_ocean_flag(sds_array, fillna=True,byte_index=0):
    """ # 3d array version [6,ix,iy]
        Assume sds_array = hdf.select('Cloud_Mask') [6,nx,ny]
         File: Cloud_Mask_1.hdf which stores first important 6bits
     +Flags
     0: 00 = water
     1: 01 = coastal
     2: 10 = desert
     3: 11 = land
    """
    def bits_stripping(bit_start,bit_count,value):
        bitmask=pow(2,bit_start+bit_count)-1
        return np.right_shift(np.bitwise_and(value,bitmask),bit_start)
    cm_array = sds_array.get()
    cm_array = np.squeeze(cm_array[:,:,byte_index])
    ncarray = np.vstack(list(map(lambda x: bits_stripping(6,7,x), cm_array))).astype(np.float64)
    if fillna:
        nan_idx = np.where(cm_array[0] == sds_array.attributes()['_FillValue'])
        ncarray[nan_idx] = np.nan
    return ncarray


def translate_const_clouds_array(patch, clouds_mask, width=128, height=128, thres=0.3, coord=(0,0)):
    """
    1 patch - 1 cloud-flags/patch.
    thres: range 0-1. ratio of clouds within the given patch
    function to return one clouds_patch and clouds-flag based on corrdinate (i,j)

    OUT:
      clouds_patch: np.array(128,128,nband)
      clouds_flag : Boolean{True; valid cloud patch, False; o.w.}  
    """
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
    return patch, clouds_flag    

def translate_const_ocean_array(patch, ocean_mask, width=128, height=128, thres=0.99, coord=(0,0)):
    """
    # new function for into_mod_normed_record 2021/01/12 for ocean flag
    1 patch - 0 ocean-flags/patch.
    thres: range 0-1. ratio of ocean within the given patch
    function to return one ocean_patch and ocean-flag based on corrdinate (i,j)
    return patch with 99% of ocean pixel

    OUT:
      ocean_patch: np.array(128,128,nband)
      ocean_flag : Boolean{True; valid cloud  patch on ocean, False; o.w.}  
    """
    i,j = coord
    # prep flag
    ocean_flag = False
    # main process
    if np.any(ocean_mask[i:i+width,j:j+height] == 0):
        tmp = ocean_mask[i:i+width,j:j+height]
        noceans = len(np.argwhere(tmp == 0))
        if noceans/(width*height) > thres:
            # valid patch
            ocean_flag = True
    return patch, ocean_flag    

def read_hdf(hdf_file, varname='varname'):
    """Read `hdf_file` and extract relevant varname .
    """
    hdf = SD(hdf_file, SDC.READ)
    return hdf.select(varname)

def proc_sds(sds_array, sdsmax=32767):
    """
    IN: array = hdf_data.select(variable_name)
    """
    array = sds_array.get()
    array = array.astype(np.float32)
    # check bandinfo
    _bands = sds_array.attributes()['band_names']
    bands = _bands.split(",")

    # error code 
    # higher value than 32767
    # C6 MODIS02 version, 65500-65534 are error code defined by algorithm
    # 65535 is _FillValue 
    # So far 65535(FillValue) is the largest value in native 16bit integer
    err_idx = np.where( (array > sdsmax) 
                  & (array < sds_array.attributes()['_FillValue']) )
    # nan process
    nan_idx = np.where( array == sds_array.attributes()['_FillValue'])
    if len(nan_idx) > 0:
        array[nan_idx] = np.nan
    else:
        pass
    if len(err_idx) > 0:
        array[err_idx] = np.nan
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
    for iband in range(len(scales)):
        scales_array[iband, :,:] = scales[iband]*offset_array[iband,:,:]
    return scales_array, bands


def aug_array(ref_array, ems_array, ref_bands=[], ems_bands=[], cref_bands=[], cems_bands=[]):
    _,nx,ny = ref_array.shape

    # ref SB
    array_list = []
    for idx, iband in enumerate(cref_bands):
        for iref_band in ref_bands:
            if iband ==  iref_band:
                array_list+=[ref_array[idx].reshape(nx,ny,1)]
    # emissive SB
    for idx, iband in enumerate(cems_bands):
        for iems_band in ems_bands:
            if iband ==  iems_band:
                array_list+=[ems_array[idx].reshape(nx,ny,1)]
    # concatenation
    return np.concatenate(array_list, axis=2)


def make_scale_mask_operator(patches, filename, coordinate, gmean, gstdv, channels=6,nsigma=2):
    """Apply [0,1] scaling and circle mask 

    """
    def get_masks(rpatch_size, channels):
        mask = np.zeros((rpatch_size, rpatch_size), dtype=np.float64)
        cv2.circle(mask, center=(rpatch_size // 2, rpatch_size // 2),
                radius=rpatch_size//2, color=1, thickness=-1)
        mask = np.expand_dims(mask, axis=-1)
        #  multiple dimension
        mask_list = [ mask for i in range(channels)]
        masks = np.concatenate(mask_list, axis=-1)
        return masks

    # scale [0,1]
    if patches.ndim == 4:
        for ichannel in range(channels):
            ulim = gmean[ichannel]+nsigma*gstdv[ichannel]
            patches[:,:,:,ichannel] = patches[:,:,:,ichannel] / ulim
    elif patches.ndim == 3:
        # NOTE: The following three lines makes bug for three-dimension code
        #for ichannel in range(channels):
        #    ulim = gmean[ichannel]+nsigma*gstdv[ichannel]
        #    patches[:,:,ichannel] = patches[:,:,ichannel] / ulim
        # NOTE: need 'float' to nsigma to be accurate computation
        ulim = gmean+float(nsigma)*gstdv
        patches = patches / ulim
    else:
      raise ValueError("Inappropriate patch shape")

    # fill 1.0 where pixel values over nsigma
    upper_index = np.where(patches > 1.000)
    patches[upper_index] = 1.000

    # just in case remove negative values
    zero_index = np.where(patches < 0.000)
    if len(zero_index[0]) > 0:
        # Report when negative pixel was detected
        print(" #### !!!Negative Pixels Are Detected!!!", flush=True)
        patches[zero_index] = 0.00

    # circle mask
    patch_size = patches.shape[1]
    if patches.ndim == 4:
        mask = get_masks(patch_size,channels).reshape(1,patch_size,patch_size,channels)
        patches = patches * mask
    elif patches.ndim == 3:
        mask = get_masks(patch_size,channels)
        patches = patches * mask
    return filename, coordinate,patches

