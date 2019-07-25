# _*_ coding: utf-8 _*_
"""
  + Desctiption
  Test code to get global mean and standard deviation from multiple files

  + Design
  1. pre-process each hdf files parallelly ( or load already computed files)
  
  mean
  2. compute sum of each hdf file and number of files
  3. mpi reduce sum info and number of files, and mpi sum them
  4. compute sum( sum(data) ) / sum( len(# of data) ) at rank 0

  std
  5. distribute the global mean to each hdf file
  6. get sum of squre residual and number of data
  7. compute sum( sum(square residual data) ) / sum( len(# of data) ) at rank 0

  output
  8. export N band' global mean and global std

  check result
  9. compare the MPI result with non-mpi version. If not same, debug
"""
# Test directory and file info
# + test directory
#   regression_laads_2000_2018_1
# + test files
#   MOD021KM.A2009044.1800.061.2017296092652.hdf
#   MOD021KM.A2011302.1950.061.2017327062615.hdf 
#

import os
import gc
import sys
import glob
import argparse
import numpy as np

from mpi4py import MPI
from pyhdf.SD import SD, SDC
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

# own library
homedir=str(Path.home())
clouds_dir="/clouds/src_analysis/lib_hdfs"
sys.path.insert(1,os.path.join(sys.path[0],homedir+clouds_dir))
from alignment_lib import gen_mod02_img, gen_mod35_img
from alignment_lib import const_clouds_array
from alignment_lib import _gen_patches
from alignment_lib import get_filepath
from lib_modis02   import load_tfrecord
from lib_modis02   import _get_imgs
from lib_modis02   import _get_num_imgs

def compute_global_single(mod02_datadir='./', mod35_datadir='./', 
                          thres=0.3, nx=128, ny=128, nbands=6):
  #TODO: add if & lines to read tfrecord files
  mod02_filelist = glob.glob(mod02_datadir+'/*.hdf')
  clouds_patches_list = []
  for m2_file in mod02_filelist:
    hdf_m2 = SD(m2_file, SDC.READ)
    m35_file = get_filepath(m2_file,mod35_datadir, prefix='MOD35_L2.A')

    # get m2 and cloud mask image process
    mod02_img = gen_mod02_img(hdf_m2)
    hdf_m35 = SD(m35_file, SDC.READ)
    clouds_mask_img = gen_mod35_img(hdf_m35)

    # valid patches
    mod02_patches = _gen_patches(mod02_img, normalization=False)
    tmp_clouds_patches_list, _ = const_clouds_array(
              mod02_patches, clouds_mask_img, thres=thres
    ) # shape [# of patches][128,128,6]
    for ipatch in tmp_clouds_patches_list:
      clouds_patches_list.append(ipatch.reshape(nx*ny, nbands))
    # clouds_patches_list [# of patches][128*128, 6]

  # global 
  alls = np.concatenate(clouds_patches_list, axis=0) 
  gmean = np.mean(alls, axis=0)
  gstdv = np.std(alls, axis=0)
  return gmean, gstdv


def compute_local(m2_filelist=[], 
                 mod35_datadir='./', 
                 thres=0.3, nx=128, ny=128, nbands=6,
                 compute_type='xxx', 
                 global_mean=np.zeros((6)), 
                 tf_filelists=[] ):
  """
    global_mean = [# of bands, ]
  """
  if tf_filelists:
    """
    lines in case load already copmuted clouds_patches_list and clouds_xy_list
    from tfrecord
    
    tf_filelists is list of tfrecord s.t. [xxx.tfrecord, yyyy.tfreocrd, ...]
    """  
    # get number of patches
    npatches=0
    for tfrecord in tf_filelists:
      npatches += _get_num_imgs(tfrecord)
    
    print(" ### Rank %d | Process %d of patches for computatoin ###"
             % ( MPI.COMM_WORLD.Get_rank() , npatches), flush=True )
    
    shape = (nx, ny, nbands)
    dataset = load_tfrecord(
                tf_filelists,
                shape,
                batch_size=1,
    )
    _clouds_patches_list = _get_imgs(dataset, n=npatches)
    clouds_patches_list = _clouds_patches_list[0]
    # _clouds_patches_list[0]: np.array data
    # _clouds_patches_list[1]: np.array shapes?
    # _clouds_patches_list[2]: original hdf filelist with abspath
    
    # collect garbages and save memory
    gc.collect()
    
    # compute against all data
    all_list = []
    for ipatch in clouds_patches_list:
      all_list.append(ipatch.reshape(nx*ny, nbands))
    alls = np.concatenate(all_list, axis=0) 
    #alls  [# of patches][128*128, 6]

    ndata=int(len(clouds_patches_list)*128*128) # number of data in one band
    # In general, all band must have same number of data(# of patches by # of pixels/patch )
    gc.collect()
    #
  else:
    #
    clouds_patches_list = []
    for m2_file in m2_filelist:
      # get hdf file
      hdf_m2 = SD(m2_file, SDC.READ)
      m35_file = get_filepath(m2_file,mod35_datadir, prefix='MOD35_L2.A')

      # get m2 and cloud mask image process
      mod02_img = gen_mod02_img(hdf_m2)
      hdf_m35 = SD(m35_file, SDC.READ)
      clouds_mask_img = gen_mod35_img(hdf_m35)

      # valid patches
      mod02_patches = _gen_patches(mod02_img, normalization=False)
      _clouds_patches_list, _ = const_clouds_array(
              mod02_patches, clouds_mask_img, thres=thres
      ) # shape [# of patches][128,128,6]
      clouds_patches_list.extend(_clouds_patches_list)
  
    # collect garbages and save memory
    gc.collect()
  
    # compute against all data
    all_list = []
    for ipatch in clouds_patches_list:
      all_list.append(ipatch.reshape(nx*ny, nbands))
    alls = np.concatenate(all_list, axis=0) 
    #alls = np.concatenate(clouds_patches_list, axis=0) 
    #alls  [# of patches][128*128, 6]

    ndata=int(len(clouds_patches_list)*128*128) # number of data in one band
    # In general, all band must have same number of data(# of patches by # of pixels/patch )
    gc.collect()


  try:
    if compute_type == 'mean':
      sums = np.sum(alls, axis=0)
      #sums = np.sum(np.concatenate(alls, axis=0), axis=0)
      return sums, ndata

    elif compute_type == 'stdv' or compute_type == 'std':
      stdv = np.sum(pow((alls- global_mean),2), axis=0) 
      #stdv = np.sum(np.concatenate(pow((alls- global_mean),2), axis=0), axis=0) 
      return stdv, ndata

  except:
    raise NameError(" compute_type was not correctly specified ") 

def get_args():
    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, description=__doc__
    )
    p.add_argument(
        "--mod02_datadir",
        type=str
    )
    p.add_argument(
        "--mod35_datadir",
        type=str
    )
    p.add_argument(
        "--tfrecord_datadir",
        type=str
    )
    p.add_argument(
        "--operate_single",
        help="operate global mean & stdv operation by single core/ Debug for No parallelized version",
        type=int,
        default=0
    )

    FLAGS = p.parse_args()
    return FLAGS

if __name__ == "__main__":

    # flags
    FLAGS = get_args()

    #------------------------------------------------------------------
    # Single version
    #------------------------------------------------------------------
    if FLAGS.operate_single == 1:
      """
        operate_single {0,1}: 0; DONOT compute. 1; DO compute
      """
      true_gmeans, true_gstds = compute_global_single(
                              FLAGS.mod02_datadir,
                              FLAGS.mod35_datadir
                              )
      if MPI.COMM_WORLD.Get_rank() == 0:
        print("Non Parallel: Global Mean : ", true_gmeans, flush=True)
        print("Non Parallel: Global Stdv : ", true_gstds,  flush=True)
        print(" Terminate computation for truth ", flush=True)
        print(" ", flush=True)
      gc.collect()

    #------------------------------------------------------------------
    # Parallel version
    #------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # get two filelists
    filelists = []
    for i, f in enumerate(glob.glob(FLAGS.mod02_datadir+'/*.hdf')):
        if i % size == rank:
            filelists.append(os.path.abspath(f))

    if not filelists:
        raise ValueError("mod02 directory does not match any files")
    
    tf_filelists = []
    for i, f in enumerate(glob.glob(FLAGS.tfrecord_datadir+'/*.tfrecord')):
        if i % size == rank:
            tf_filelists.append(os.path.abspath(f))

    if not filelists:
      if not tf_filelists:
        raise ValueError("mod02 directory and tfrecord directory do not match any files")
        

    #
    sums, ndata = compute_local(m2_filelist=filelists, 
                                mod35_datadir=FLAGS.mod35_datadir,
                                compute_type='mean',
                                tf_filelists=tf_filelists
    )
    print("Rank %d done." % rank, flush=True)

    comm.Barrier() # wait here to syncronize all process
    
    # get globalsums
    global_sums = comm.gather(
      sums,
      root=0
    )
    global_ndata = comm.gather(
      ndata,
      root=0
    )
        
    global_mean = None
    if rank == 0:
      # compute mean
      _global_mean = np.asarray(global_sums)
      global_mean = np.sum(_global_mean, axis=0)/np.sum(global_ndata)
      print("Parallel: Global Mean : ", global_mean, flush=True)
    
      # save data here
    else: 
      # prep recvbuf TODO: let 6 be a argument as number of bands
      #computed_global_mean = np.empty(6,dtype='float64')
      global_mean = np.empty(6,dtype='float64')

    # scatter global mean
    #comm.Scatter(global_mean,computed_global_mean, root=0)
    comm.Bcast(global_mean, root=0)
    gc.collect()

    # compute standard deviation
    # global_mean; np.ndarray [ # of bands ]        
    comm.Barrier()      
    res_sums, ndata = compute_local(m2_filelist=filelists, 
                                    mod35_datadir=FLAGS.mod35_datadir,
                                    compute_type='stdv',
                                    global_mean=global_mean,
                                    tf_filelists=tf_filelists
    )
    
    print("Rank %d done." % rank, flush=True)

    comm.Barrier() # wait here to syncronize all process
    
    # get globalsums
    global_res_sums = comm.gather(
      res_sums,
      root=0
    )
    global_ndata = comm.gather(
      ndata,
      root=0
    )
        
    if rank == 0:
      # compute mean
      #print("global_res_sums", global_res_sums)
      _global_res_sums = np.asarray(global_res_sums)
      global_stdv = np.sqrt(np.sum(_global_res_sums, axis=0)/np.sum(global_ndata))
      print("Parallel: Global Stdv : ", global_stdv, flush=True)

