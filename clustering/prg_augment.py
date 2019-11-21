# _*_  coding : utf-8  _*_
#
# + Add normed flag
#

import os
import re
import sys
import glob
import time
import argparse
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from pyhdf.SD import SD, SDC

libdir="/home/tkurihana/Research/clouds/src_analysis/lib_hdfs"
sys.path.insert(1,os.path.join(sys.path[0],libdir))
from alignment_lib import _decode_cloud_flag, decode_cloud_flag,  const_clouds_array

sys.path.insert(1,os.path.join(sys.path[0],libdir))
from analysis_lib import mod06_proc_sds, _gen_patches

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
  "--prefix",
  type=str,
  default='MOD35_L2.A',
  help=' Prefix for mod35'
)
p.add_argument(
  "--cloud_thres",
  type=float,
  default=0.3,
)
p.add_argument(
  "--model_dir",
  type=str,
  default='/project2/foster/clouds/output/mod02/m2_02_global_2000_2018_band28_29_31',
)
p.add_argument(
  "--step",
  type=int,
  default=100000,
)
p.add_argument(
  "--outputdir",
  type=str,
  default='./output',
)
p.add_argument(
  "--output_filename",
  type=str,
  default='agument_m2_m35',
)
# + Additional line for normed version
p.add_argument(
  "--normed",
  type=int,
  default=0,
  help='IF normalization model is used +1, other number s.t. none-normed is 0',
)
p.add_argument(
  "--stats_datadir",
  type=str,
  default='./',
)
args = p.parse_args()


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

def mod02_proc_sds_mosaic(sds_array):
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

def gen_mod02_img_mosaic(
      hdf_datadir='hdf data directory',
      date='2015001'
    ):

    #ad-hoc bands assumed 6,7,20,28,29 and 31
    #print(date, flush=True)
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
      refsb_array.append(mod02_proc_sds_mosaic(refsb_sds))
    refsb_bands = [6,7]

    # band Emissive
    ev_array = []
    for ifile in ev_list:
      ev_hdf = SD(ifile, SDC.READ)
      ev_sds = ev_hdf.select("EV_1KM_Emissive")
      ev_array.append(mod02_proc_sds_mosaic(ev_sds))
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

def gen_mod35_img_mosaic(
      hdf_datadir='hdf data directory',
      date='2015001'
    ):
    mod35_file = glob.glob(hdf_datadir+'/MOD35_L2.A'+date+'.mosaic.061*.Cloud_Mask_1.hdf')[0]
    hdf    = SD(mod35_file, SDC.READ)
    cm_sds = hdf.select('Cloud_Mask')
    clouds_mask_array = decode_cloud_flag(cm_sds)
    return clouds_mask_array

def get_filepath(filepath, datadir, prefix='', mosaic=True):
    """filepath for another modis data corresponding given filepath
    """
    #FIXME
    #date = os.path.basename(filepath)[10:22] # ex. 2017213.2355
    bname = os.path.basename(filepath) # ex. 2017213.2355
    if mosaic :
      dateinfo = re.findall('[0-9]{7}.mosaic' , bname)
      date     = dateinfo[0].rstrip("['").lstrip("']")
      filelist = glob.glob(datadir+'/'+prefix+'*'+date+'.Cloud_Mask_1*.hdf')
    else  :
      dateinfo = re.findall('[0-9]{7}.[0-9]{4}' , bname)
      date     = dateinfo[0].rstrip("['").lstrip("']")
      filelist = glob.glob(datadir+'/'+prefix+'*'+date+'*.hdf')
    try:
      if len(filelist) > 0:
        # in the case file-exist
        return filelist[0]
    except:
      efile = datadir+'/'+prefix+date+'*.hdf'
      print(" Program will be forcibly terminated", flush=True)
      raise NameError(" ### File Not Found: No such file or directory "+str(efile)+" ### ")
    ofilelist = glob.glob(datadir+'/'+prefix+date+'*.hdf')
    if len(ofilelist) >0:
      return  ofilelist[0]
    else :
      return []

def get_dateFromMODIS(modisfile, mosaic=True):
    bname = os.path.basename(modisfile) # ex. 2017213.mosaic
    if mosaic :
      dateinfo = re.findall('[0-9]{7}.mosaic' , bname)
      date     = dateinfo[0].rstrip("['").lstrip("']").strip(".mosaic")
    else  :
      dateinfo = re.findall('[0-9]{7}.[0-9]{4}' , bname)
      date     = dateinfo[0].rstrip("['").lstrip("']")
    return date

def compute_augment(encoder, m2_file, thres=0.3, height=128, width=128, bands=6, mosaic=True): 
    """
    """ 
    if mosaic :
      # get mod02 dirname
      m2_file_dir = os.path.dirname(os.path.abspath(m2_file))

      # get corresponding filepath
      m35_file = get_filepath(m2_file, args.mod35_datadir, prefix='MOD35_L2.A')
      # TODO: add correct exit line for mpi4py; otherwise mpi4py does not shut down
      if len(m35_file) == 0:
        print('NO DATA', flush=True)
        exit(0);

      # hdf mod35
      if not os.path.exists(m35_file):
        print(os.path.basename(m35_file))

      # gen_mod35_img_mosaic wrap this method
      #hdf_m35 = SD(m35_file, SDC.READ)
      # get current processing file date
      cdate = get_dateFromMODIS(m2_file)
      # get image process from mosaic file
      mod02_img = gen_mod02_img_mosaic(m2_file_dir, date=cdate)
      m35_file_dir = os.path.dirname(os.path.abspath(m35_file))
      clouds_mask_img = gen_mod35_img_mosaic(m35_file_dir, date=cdate)
      #
      # =============================================================
    else :
      # =============================================================
      #
      # hdf mod02
      hdf_m2 = SD(m2_file, SDC.READ)

      # get corresponding filepath
      m35_file = get_filepath(m2_file, args.mod35_datadir, prefix='MOD35_L2.A')
      # TODO: add correct exit line for mpi4py; otherwise mpi4py does not shut down
      if len(m35_file) == 0:
        print('NO DATA', flush=True)
        exit(0);

      # hdf mod35
      if not os.path.exists(m35_file):
        print(os.path.basename(m35_file))
      hdf_m35 = SD(m35_file, SDC.READ)
      # get image process
      mod02_img = gen_mod02_img(hdf_m2)
      clouds_mask_img = gen_mod35_img(hdf_m35)

    # z score normlization; mean 0, stdv +- 1
    if args.normed == 1:
      if MPI.COMM_WORLD.Get_rank() == 0:
        print("###  Normalization is applied ###")
      global_mean = np.load(glob.glob(args.stats_datadir+'/*_gmean.npy')[0])
      global_stdv = np.load(glob.glob(args.stats_datadir+'/*_gstdv.npy')[0])
      mod02_img -= global_mean
      mod02_img /= global_stdv   
    

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

def save_as_file(encs_mean, clouds_xy_list, mod_file, outputdir='./', filename=''):
      rank = MPI.COMM_WORLD.Get_rank()
      filepath = mod_file
      # name 
      dtime    = os.path.basename(filepath)[10:22] #2017213.2355
      os.makedirs(outputdir, exist_ok=True)
      #
      npatch = int(encs_mean.shape[0])
      np.savez(outputdir+'/'+filename+'_'+dtime+'_'+str(rank), 
             encs_mean=encs_mean, 
             clouds_xy=np.asarray(clouds_xy_list),
      )
      oname = filename+'_'+dtime+'_'+str(rank)+'.npz'
      print("  save encs_mean: %s  Rank: %d  %d patches" % (oname, rank,  npatch) , flush=True)


if __name__ == "__main__":
  # File load
  # mosaic --> has to specify one file then compute different file parallelly
  filelists = glob.glob(args.mod02_datadir+'/*EV_1KM_Emissive_1.hdf')
  # normal
  #filelists = glob.glob(args.mod02_datadir+'/*.hdf')
  filelists.sort()
  print(" Number of Total Files ==", len(filelists), flush=True)

  #================================================
  # MPI Parallelization 
  #================================================

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  print("\n ### MPI rank {} |  MPI size {} ### \n".format(rank, size) , flush=True)
  # Allocate Files in each node
  filelist = []
  for fidx, ifile in enumerate(filelists):   
    if fidx % size == rank :
      filelist.append(os.path.abspath(ifile))
  
  print(len(filelist), rank)

  # debug
  #time.sleep(10)
  #exit(0)

  # model load
  model_dir = args.model_dir
  step = args.step
  # model
  encoder_def = model_dir+'/encoder.json'
  encoder_weight = model_dir+'/encoder-'+str(step)+'.h5'
  with open(encoder_def, "r") as f:
    encoder = tf.keras.models.model_from_json(f.read())
  encoder.load_weights(encoder_weight)

  # computation
  for idx, filename in enumerate(filelist):
      #if not int(os.path.basename(filename)[10:14]) == 2000:
      # augmentation process here
      clouds_patches, clouds_xy_list, m2_file  = compute_augment(
        encoder, filename, thres=args.cloud_thres, height=128, width=128, bands=6, mosaic=True
      )
      # save clouds_info = (encs, xy_list, filenames)
      print(len(clouds_patches))
      if not len(clouds_patches) == 0:
        save_as_file(clouds_patches, clouds_xy_list, m2_file, 
               outputdir=args.outputdir, filename=args.output_filename)

  print("Rank %d done." % rank, flush=True)
