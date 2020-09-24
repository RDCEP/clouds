#
import os
import re
import cv2
import sys
import json
import glob
import copy
import time
import pickle
import itertools
import numpy as np
import pandas as pd
import scipy as sc
from scipy.stats import mode
from pyhdf.SD import SD, SDC
import tensorflow as tf
from tensorflow.contrib.data import parallel_interleave

# specific metrics
from scipy.stats import wasserstein_distance as EMD

def data_extractor_resize_fn(filelist,prefetch=1,height=32,width=32,channel=6,read_threads=4, distribute=(1, 0)):
    def parser(ser):
        """
        Decode & Pass datast in tf.record
        *Cuation*
        floating point: tfrecord data ==> tf.float64
        """
        #"patch": tf.FixedLenFeature([], tf.string),
        features = {
            "shape": tf.FixedLenFeature([2], tf.int64),
            "filename": tf.FixedLenFeature([], tf.string),
            "coordinate": tf.FixedLenFeature([2], tf.int64),
            "Cloud_Optical_Thickness": tf.FixedLenFeature([], tf.string),
            "cloud_top_pressure_1km": tf.FixedLenFeature([], tf.string),
            "Cloud_Effective_Radius": tf.FixedLenFeature([], tf.string),
            "Cloud_Water_Path": tf.FixedLenFeature([], tf.string),
            "Cloud_Phase_Infrared_1km": tf.FixedLenFeature([], tf.string),
        }
        decoded = tf.parse_single_example(ser, features)
        #patch = tf.reshape(
        #    tf.decode_raw(decoded["patch"], tf.float64), decoded["shape"]
        #)
        # conversion of tensor
        #patch = tf.cast(patch, tf.float32)
        #patch = tf.image.resize_images(patch, (height, width))

        ### decode all
        cot = tf.reshape(
            tf.decode_raw(decoded["Cloud_Optical_Thickness"], tf.float64), decoded["shape"]
        )
        ctp = tf.reshape(
            tf.decode_raw(decoded["cloud_top_pressure_1km"], tf.float64), decoded["shape"]
        )
        cer = tf.reshape(
            tf.decode_raw(decoded["Cloud_Effective_Radius"], tf.float64), decoded["shape"]
        )
        cwp = tf.reshape(
            tf.decode_raw(decoded["Cloud_Water_Path"], tf.float64), decoded["shape"]
        )
        cpf = tf.reshape(
            tf.decode_raw(decoded["Cloud_Phase_Infrared_1km"], tf.float64), decoded["shape"]
        )
        
        return cot,ctp,cer,cwp,cpf
    
    dataset = (
        tf.data.Dataset.list_files(filelist, shuffle=True)
            .shard(*distribute)
            .apply(
            parallel_interleave(
                lambda f: tf.data.TFRecordDataset(f).map(parser),
                cycle_length=read_threads,
                sloppy=True,
            )
        )
    )
    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()
    #patches_list = []
    cots,ctps,cers,cwps,cpfs = [],[],[],[],[]
    with tf.Session() as sess:
        try:
            while True:
                cot,ctp,cer,cwp,cpf = sess.run(next_element)
                
                # append
                cots.append(cot)
                ctps.append(ctp)
                cers.append(cer)
                cwps.append(cwp)
                cpfs.append(cpf)
                
        except tf.errors.OutOfRangeError:
            print("OutOfRage --> finish process")
            pass
    return cots,ctps,cers,cwps,cpfs

def list2np(patches_list, height=128, width=128, channel=1, npatches=2000):
    patches = np.concatenate(
      [np.expand_dims(i, axis=0).reshape(1,height,width, channel) for i in patches_list[0]],
    axis=0)
    print("PATCH SHAPE", patches.shape)
    return patches

def load_dataset(datadir=None, filebasename='*2-10*.tfrecord', 
                 height=32, width=32,channel=6,nfiles = 1,key_list=[].
                 ):

    filelist = glob.glob(os.path.join(datadir, filebasename))
    ## load 
    cots = []
    ctps = []
    cers = []
    cwps = []
    cpfs = []
    for ifile in filelist:
      cot,ctp,cer,cwp,cpf = data_extractor_resize_fn([ifile] )
      # append
      cots.append(cot)
      ctps.append(ctp)
      cers.append(cer)
      cwps.append(cwp)
      cpfs.append(cpf)

    #### get patch
    # - original -
    #cot_patches = list2np(cots)
    #ctp_patches = list2np(ctps)
    #cer_patches = list2np(cers)
    #cwp_patches = list2np(cwps)
    #cpf_patches = list2np(cpfs)

    # - let them in dict form - 
    patches = {}
    #key_list = [
    #    'optical thickness',
    #    'phase',
    #    'top pressure',
    #    'effective radius',
    #    'water path',
    #]
    print(" Key for MOD06 PATCHES == ",key_list, flush=True)
    for ikey, mod06_list in zip(key_list, [cots, cpfs, ctps, cers, cwps]):
      patches[ikey] = list2np(mod06_list)
    print("NORMAL END")
    return patches

def pkl_loader(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load_clustering_fn(basedir=None,expname=None,cexpname=None,nclusters=-1, clf_key='HAC',):
    # specifiy datadir
    datadir = os.path.join(
        basedir, 
        '{}/nclusters-{}/{}'.format(str(expname),str(nclusters),clf_key ))

    # specifiy datadir + filename
    filename = os.path.join(datadir,f"original-hac_{cexpname}.pkl")
    
    label = pkl_loader(filename)
    return label

def build_params(patches,label=None, nclusters=-1, build_all=False):
    """ patches[n,128,128]"""
    hist_list = []
    print(f"Build All = {build_all}", flush=True)
    for icluster in range(nclusters):
        cdx = np.where(label == icluster)
        n, h,w = patches[cdx]
        # vectorize axis=1
        if build_all:
          # True: get entire distribution
          tmp = patches.reshape(n*h*w)
          ok_idx = np.where(~np.isnan(tmp))
          hist_list.append(tmp[ok_idx])
        else:
          # False: get individual distributino
          tmp = patches.reshape(n,h*w)
          ok_idx = np.where(~np.isnan(tmp))
          hist_list.append(tmp[ok_idx])
    return hist_list


def compare_hist_emd(hist1, hist2):
   """
    Cite: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
   """ 
   # hist1 and hist 2 should be 1 dimensional vector
   return EMD(hist1, hist2)


def compare_hist_opencv(hist1, hist2, metric=None):
    """
      Cite: https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist#cv2.compareHist
            https://tat-pytone.hatenablog.com/entry/2019/03/07/220938
    """
    return cv2.compareHist(hist1, hist2, metric)

def run_intra_similarity_fn(patch=None, label=None, nlcusters=None,nstep=100, density=True):
    """
    """
    stime = time.time()
    # normalize [0,1*nstep]
    nmin = np.nanmin(patch)
    nmax = np.nanmax(patch)
    patch= (patch-nmin)/(nmax-nmin)*nstep)

    # build vectorized data for each cluster
    # type: List
    all_data = build_params(patch,label, nclusters, build_all=True)

    # compute histogram
    results = {}
    for tmp, cluster in zip(all_data, range(nclusters)):
      hists = {}
      tmp_results = {}

      # compute histogram for each patch
      for idx, data in enumerate(tmp):
        hist_inv, _  = np.histogram(data, np.linspace(0,nstep, nstep+1),density=density)
        hists[idx] = hist_inv

      # compute similarity across patches within same cluster
      x = [i for i in range(tmp.shape[0])]
      for i,j in itertools.combinations(x,2):
        hist1 = hists[i]
        hist2 = hists[j]
        tmp = {}
        tmp['emd'] = compare_hist_emd(hists1,hists2)

        # opencv
        metric_list = [
          'correlation','chi-square','intersection','bhattacharyya'
        ]
        metrics ={
          'correlation': cv2.HISTCMP_CORREL,
          'chi-square' : cv2.HISTCMP_CHISQR,
          'intersection' : cv2.HISTCMP_INTERSECT,
          'bhattacharyya' : cv2.HISTCMP_BHATTACHARYYA
        }
      
        for metic in metric_list:
          tmp[f"{metric}"] = compare_hist_opencv(hist1, hist2, metrics[metric])
        
        tmp_results[f"{i}-{j}"] = tmp
    
      # wrap another dict
      results[f'ncluster-{ncluster}'] = tmp_results

    etime = time.time()
    elapse = (etime - stime)/60
    print(f"ANALYSIS NORMAL TERMINATE {elapse} min", flush=True)
    return results

def run_inter_similarity_fn(patch=None, label=None, nlcusters=None,nstep=100, density=True):
    """
    """
    stime = time.time()
    # normalize [0,1*nstep]
    nmin = np.nanmin(patch)
    nmax = np.nanmax(patch)
    patch= (patch-nmin)/(nmax-nmin)*nstep)

    # build vectorized data for each cluster
    # type: List
    data = build_params(patch,label, nclusters, build_all=True)

    # compute histogram
    hists = {}
    for tmp, cluster in zip(data, range(nclusters)):
      hist_alls, _  = np.histogram(tmp, np.linspace(0,nstep, nstep+1),density=density)
      hists[cluster] = hist_alls

    # inter cluster 
    results = {}
    x = [i for i in range(nclusters)]
    for i,j in itertools.combinations(x,2):
      hist1 = hists[i]
      hist2 = hists[j]
      tmp = {}
      tmp['emd'] = compare_hist_emd(hists1,hists2)

      # opencv
      metric_list = [
        'correlation','chi-square','intersection','bhattacharyya'
      ]
      metrics ={
        'correlation': cv2.HISTCMP_CORREL,
        'chi-square' : cv2.HISTCMP_CHISQR,
        'intersection' : cv2.HISTCMP_INTERSECT,
        'bhattacharyya' : cv2.HISTCMP_BHATTACHARYYA
      }
      
      for metic in metric_list:
        tmp[f"{metric}"] = compare_hist_opencv(hist1, hist2, metrics[metric])
        
      results[f"cluster{i}-cluster{j}"] = tmp
    
    etime = time.time()
    elapse = (etime - stime)/60
    print(f"ANALYSIS NORMAL TERMINATE {elapse} min", flush=True)
    return results


def run_inter_ratio_fn(patch, label, nclusters=None,ncategory=4):
    """ Compute ratio of categories across class
    """
    # build vectorized data for each cluster
    # type: List
    stime = time.time()
    data = build_params(patch,label, nclusters, build_all=True)
      
    inter_results = {}
    for cluster, tmp in enumerate(data):
      tmp_result = {}
      for cat in range(ncategory):
        if cat < 3:
          tmp_result[cat] = len(np.where(tmp == cat)[0])/tmp.size*100
        elif cat>=3:
          tmp_result[cat] = len(np.where(tmp >= cat)[0])/tmp.size*100
      inter_results[f'ncluster-{cluster}'] = tmp_result
    etime = time.time()
    elapse = (etime - stime)/60
    print(f"ANALYSIS NORMAL TERMINATE {elapse} min", flush=True)
    return inter_results
          

def run_intra_ratio_fn(patch, label, nclusters=None,ncategory=4):
    """ Compute ratio of categories across same class
    """
    # build vectorized data for each cluster
    # type: List
    stime = time.time()
    all_data = build_params(patch,label, nclusters, build_all=False)
      
    intra_results = {}
    for cluster, data in enumerate(all_data):
      inter_result = {}
      for idx, tmp in enumerate(data):
        for cat in range(ncategory):
          if cat < 3:
            tmp_result[cat] = len(np.where(tmp == cat)[0])/tmp.size*100
          elif cat>=3:
            tmp_result[cat] = len(np.where(tmp >= cat)[0])/tmp.size*100
        inter_results[idx] = tmp_result
      intra_results[f"ncluster-{ncluster}"] = inter_results
    etime = time.time()
    elapse = (etime - stime)/60
    print(f"ANALYSIS NORMAL TERMINATE {elapse} min", flush=True)

    return intra_results


def get_argument(verbose=True):
    p = argparse.ArgumentParser()
    p.add_argument('--tf_datadir', type=str, default=None)
    p.add_argument('--output_basedir', type=str, default=None)
    p.add_argument('--model_datadir', type=str, default=None)
    p.add_argument('--cache_datadir', type=str, default=None)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--channel', type=int, default=None)
    p.add_argument('--expname', type=str, default=None)
    p.add_argument('--cexpname', type=str, default=None)
    p.add_argument('--copy_size', type=int, default=None)
    p.add_argument('--nclusters', type=int, default=None)
    p.add_argument('--alpha', type=int, default=None, help='coef to divide large patches per every alpha subsets')
    p.add_argument('--clf_key', type=str, default=None)
    p.add_argument('--full_tree', action='store_true')
    args = p.parse_args()
    if verbose:
      for f in args.__dict__:
        print("\t", f, (25 - len(f)) * " ", args.__dict__[f], flush=True)
      print("\n",flush=True)
    return args


if __name__ == "__main__":
    # create argument parser
    FLAGS = get_argument(verbose=True)

    key_list = [
        'optical thickness',
        'phase',
        'top pressure',
        'effective radius',
        'water path',
    ]
  
    # get mod06 patches from holdlout tfrecord
    # type: dict
    patches = load_dataset(datadir=FLAGS.tf_datadir, 
                filebasename='0-0_mod06_tf2-10.tfrecord', key_list=key_list,
                height=FLAGS.height, width=FLAGS.width,channel=FLAGS.channel)

    # load clustering result
    label = load_clustering_fn(basedir=FLAGS.cluster_datadir,
                                expname=FLAGS.expname, cexpname=FLAGS.cexpname,
                                nclusters=FLAGS.nclusters
                                ) 


    
    ### save 
    outputdir = os.path.join(
        FLAGS.output_basedir, 
        '{}/nclusters-{}/{}'.format(str(FLAGS.expname),str(FLAGS.nclusters), FLAGS.clf_key ))
    os.makedirs(outputdir, exist_ok=True)

    for key in key_list:
      print(f" START ANALYZE {key} \n", flush=True)

      if key != 'phase':
        patch = patches[ikey]

        # compute inter cluster similarity
        inter_result = run_inter_similarity_fn(patch, label, nlcusters=FLAGS.nclusters, 
                                             nstep=FLAGS.nstep, density=True):

        ## scores: dump to pickle
        with open(os.path.join(outputdir, f'sim-inter_{FLAGS.cexpname}_{key}_{FLAGS.nstep}.pkl'), 'wb') as f:
          pickle.dump(inter_results, f )
        print("NORMAL END : INTER")


        # compute intra cluster similarity
        intra_result = run_intra_similarity_fn(patch, label, nlcusters=FLAGS.nclusters, 
                                             nstep=FLAGS.nstep, density=True):
        ## scores: dump to pickle
        with open(os.path.join(outputdir, f'sim-intra_{FLAGS.cexpname}_{key}_{FLAGS.nstep}.pkl'), 'wb') as f:
          pickle.dump(inter_results, f )
        print("NORMAL END : INTRA")


      elif key == 'phase':
        # phase is categorical data; four category; 0:clear sky, 1:liquie, 2:ice, 3~:undef
        """Approach: Compute ratio of four categories
        """
        # inter
        inter_result = run_inter_ratio_fn(patch, label, nclusters=FLAGS.nclusters,ncategory=4)
        ## scores: dump to pickle
        with open(os.path.join(outputdir, f'sim-inter_{FLAGS.cexpname}_{key}_{FLAGS.nstep}.pkl'), 'wb') as f:
          pickle.dump(inter_results, f )
        print("NORMAL END : INTER")

        # intra
        intra_result = run_inter_ratio_fn(patch, label, nclusters=FLAGS.nclusters,ncategory=4)
        ## scores: dump to pickle
        with open(os.path.join(outputdir, f'sim-intra_{FLAGS.cexpname}_{key}_{FLAGS.nstep}.pkl'), 'wb') as f:
          pickle.dump(inter_results, f )
        print("NORMAL END : INTRA")






