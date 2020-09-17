# Classification for clustering
# Coded by Takuya Kurihana Sep 14 2020
#
author = "tkurihana@uchicago.edu"

import os
import gc
import sys
import cv2
import math
import time
import json
import copy
import glob
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.data import parallel_interleave

#version1
from sklearn.cluster import AgglomerativeClustering as HAC
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARS
from sklearn.metrics import completeness_score as CLS
from sklearn.metrics import  homogeneity_score as HS

def data_extractor_resize_fn(filelist,prefetch=1,height=32,width=32,channel=6,read_threads=4, distribute=(1, 0)):
    def parser(ser):
        """
        Decode & Pass datast in tf.record
        *Cuation*
        floating point: tfrecord data ==> tf.float64
        """
        features = {
            "shape": tf.FixedLenFeature([3], tf.int64),
            "patch": tf.FixedLenFeature([], tf.string),
            "filename": tf.FixedLenFeature([], tf.string),
            "coordinate": tf.FixedLenFeature([2], tf.int64),
        }
        decoded = tf.parse_single_example(ser, features)
        patch = tf.reshape(
            tf.decode_raw(decoded["patch"], tf.float64), decoded["shape"]
        )
        # conversion of tensor
        patch = tf.cast(patch, tf.float32)
        patch = tf.image.resize_images(patch, (height, width))
        return patch
    
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
    patches_list = []
    with tf.Session() as sess:
        try:
            while True:
                patch = sess.run(next_element)
                patches_list.append(patch)
        except tf.errors.OutOfRangeError:
            print("OutOfRage --> finish process")
            pass
    return patches_list

def load_latest_model(model_dir, mtype):
    #TODO add restart model dir and restart argument?
    latest = 0, None
    # get trained wegiht 
    for m in os.listdir(model_dir):
        if ".h5" in m and mtype in m:
            epoch = int(m.split("-")[1].replace(".h5", ""))
            latest = max(latest, (epoch, m))

    epoch, model_file = latest

    if not os.listdir(model_dir):
        raise NameError("no directory. check model path again")

    print(" Load {} at {} epoch".format(mtype, epoch))
    model_def = model_dir+'/'+mtype+'.json'
    model_weight = model_dir+'/'+mtype+'-'+str(epoch)+'.h5'
    with open(model_def, "r") as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(model_weight)
    return model

def rotate_fn(images, angles):
    """
    images : 4d tensor [batch, height, width channel]
      original oprion - nearest. Additional: biilnear
    """
    rotated_tensor_images = tf.contrib.image.transform(
      images,
      tf.contrib.image.angles_to_projective_transforms(
        angles, tf.cast(tf.shape(images)[1], tf.float32), 
            tf.cast(tf.shape(images)[2], tf.float32)
        ),
    )
     #interpolation='BILINEAR'
    return rotated_tensor_images

def load_dataset(datadir=None, filebasename='2-10*.tfrecord', 
                 height=32, width=32,channel=6,nfiles = 1):

    filelist = glob.glob(os.path.join(datadir, filebasename))
    ## load 
    fdx = np.random.randint(0,len(filelist),nfiles)
    patches_list = []
    for ifile in [filelist[i] for i in fdx]:
      patches_list.append(data_extractor_resize_fn([ifile],height=height,width=width,channel=channel ))
    print("NORMAL END")

    ## get patch
    patches = np.concatenate(
      [np.expand_dims(i, axis=0).reshape(1,height,width, channel) for i in patches_list[0]],
    axis=0)
    print("PATCH SHAPE", patches.shape)
    return patches

def make_pairdataset(patches, idx1=None, idx2=None):
    #
    spatches = np.concatenate(
      [np.expand_dims(patches[idx1],axis=0), np.expand_dims(patches[idx2], axis=0) ], 
    axis=0)
    return spatches

def copy_rot_fn(patches, height=None, width=None, ch=None, copy_size=None):
    img_list = []
    for patch in patches:
        img_list.extend([np.reshape(patch, (1,height,width,ch))])
        img_list.extend([ np.expand_dims(np.copy(patch.reshape(height,width,ch)), axis=0) 
                     for i in range(copy_size-1)])
    imgs = np.concatenate(img_list, axis=0)
    print(imgs.shape)
    
    radians = []
    for j in range(patches.shape[0]):
        radians.extend([i*math.pi/180 for i in np.linspace(0,360,copy_size+1) ][:-1] )
    print(len(radians))
    rimgs_tf = rotate_fn(imgs, angles=radians)
    rtest_imgs = tf.keras.backend.eval(rimgs_tf)
    del imgs, rimgs_tf
    return rtest_imgs


def compute_hac(encoder, clf=None,spatches=None,):
    encs = encoder.predict(spatches)
    del spatches
    gc.collect()
    n,h,w,c = encs.shape
    #clabels = clf.fit_predict(encs.reshape(n, h*w*c))
    clustering = clf.fit(encs.reshape(n, h*w*c))
    clabels = clustering.labels_
    return clabels, clustering

def get_masks(rpatch_size, channels):

    mask = np.zeros((rpatch_size, rpatch_size), dtype=np.float64)
    cv2.circle(mask, center=(rpatch_size // 2, rpatch_size // 2),
                radius=rpatch_size//2, color=1, thickness=-1)
    mask = np.expand_dims(mask, axis=-1)
    #  multiple dimension
    mask_list = [ mask for i in range(channels)]
    masks = np.concatenate(mask_list, axis=-1)
    return masks


def left_out_fn(patches, min_std=0.08, max_mean=0.9, min_mean=0.1):
    """  Remove unsuitable patches  """

    ## NaN Mask
    width = height = patch_size = patches.shape[1]
    nan_mask = get_masks(patch_size, 1).reshape(width,height)
    nan_idx = np.where(nan_mask == 0) # (array([x,x,x,x,]) , array([y,y,y,y,y,]))
    nan_mask[nan_idx] = np.nan #
    stats_array = np.zeros((patches.shape[0], 2))

    # Extract stats info
    for i in range(patches.shape[0]):
      stats_array[i,0] = np.nanmean(patches[i,:,:,0]*nan_mask) 
      stats_array[i,1] = np.nanstd(patches[i,:,:,0]*nan_mask)

    # Compute number of patches
    idx1 = np.where(stats_array[:,0] > min_mean)
    tmp1 = stats_array[idx1]
    idx2 = np.where(tmp1[:,0] < max_mean)
    spatch_stats_array = tmp1[idx2]
    idx3 =  np.where(spatch_stats_array[:,1] > min_std )
    print(" # of  Processed Patches == ", len(idx3[0]))

    try:
      spatches =  patches[idx3]
    except Exception as e:
      print(e)

    return spatches

def eval_fn(labels=None, clabels=None, scoring=None):
    #all_scores = {}
    scores = {}
    for ikey in scoring:
      if ikey == 'ami':
        tmp = AMI(label, clabels)
      elif ikey == 'nmi':
        tmp = NMI(label, clabels)
      elif ikey == 'cls':
        tmp = CLS(label, clabels)
      elif ikey == 'hs':
        tmp = HS(label, clabels)
      elif ikey == 'ars':
        tmp = ARS(label, clabels)
      scores[ikey] = tmp
    # store as dict
    return scores

def rot_fn(patches, method=None, theta=180):
    if method == 'random':
        radians = [ itheta*math.pi/180 for itheta in np.random.randint(0,359,patches.shape[0]) ]
    elif method == 'fix':
        radians = [ theta*math.pi/180 for j in range(patches.shape[0])]        
                   
    rpatches = rotate_fn(patches, angles=radians) 
    rpatches_np = tf.keras.backend.eval(rpatches)
    return rpatches_np


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
        print("\t", f, (25 - len(f)) * " ", args.__dict__[f])
      print("\n")
    return args

if __name__ == "__main__":

    # FIXME set hardwritten param for less memory
    #npatches = 300

    # create argument parser
    FLAGS = get_argument(verbose=True)

    # get patches from holdlout tfrecord
    #patches = load_dataset(datadir=FLAGS.tf_datadir, filebasename='2-10*.tfrecord', 

    """ Modify here to leave out patches with less gradation """
    #_patches = load_dataset(datadir=FLAGS.tf_datadir, filebasename='2-10*.tfrecord', 
    _patches = load_dataset(datadir=FLAGS.tf_datadir, filebasename='2-10*.tfrecord', 
                height=FLAGS.height, width=FLAGS.width,channel=FLAGS.channel)

    #patches = left_out_fn(_patches, min_std=0.08, max_mean=0.9, min_mean=0.1)
    #idx3  = np.load('./index_2-10_normed.npy') # index satsfying criteria
    #patches = np.squeeze(_patches[idx3])[:npatches]
    #patches = patches[:npatches]

    ## large_hac6
    index_list = []
    with open('./select_index-2-10.txt') as f:
      lines = f.read()
      for line in lines.split("\n"):
        if len(line) > 0:
          index_list.append(int(line))
    sindex = np.array(index_list)
    patches = _patches[sindex]
  

    """ Original """
    # classifier load model
    model_dir = os.path.join(FLAGS.model_datadir,str(FLAGS.expname) )
    encoder = load_latest_model(model_dir, mtype='encoder')

    # score metrics
    scoring = [
        'ami',
        'nmi',
        'cls',
        'hs',
        'ars',
    ]

    # model metrics
    os.makedirs(FLAGS.cache_datadir, exist_ok=True)
    clf_dict = {
      'HAC': HAC(n_clusters=FLAGS.nclusters,memory=FLAGS.cache_datadir,compute_full_tree=FLAGS.full_tree) ,
    }
    # define a model selected from metrics
    clf = clf_dict[FLAGS.clf_key]

    ### large_hac6
    
    ### compute model 
    #label, oclustering = compute_hac(encoder, clf, patches)
    ### SAVE
    #outputdir = os.path.join(
    #    FLAGS.output_basedir, 
    #    '{}/nclusters-{}/{}'.format(str(FLAGS.expname),str(FLAGS.nclusters), FLAGS.clf_key ))
    #os.makedirs(outputdir, exist_ok=True)
    ### original models: dump to pickle
    #with open(os.path.join(outputdir, f'original-hac_{FLAGS.cexpname}.pkl'), 'wb') as f:
    #    pickle.dump(oclustering, f )
    #print("NORMAL END : CLUSTERING")
    
    #""" Process start"""
    #thetas =  np.linspace(0,360,FLAGS.copy_size+1)[:-1] 
    #all_scores = {}
    #all_labels = {}
    #for idx, theta in enumerate(range()):    
    #  rpatches = rot_fn(patches,method='fix',theta=theta)
    #  clabels, rclustering = compute_hac(encoder, clf, rpatches)
    #  gc.collect()
    #  
    #  # return dict
    #  scores = eval_fn(label, clabels, scoring=scoring)
    #  all_scores[f'nclusters-{FLAGS.nclusters}_theta-{theta}'] = scores
    #  all_labels[f'nclusters-{FLAGS.nclusters}_theta-{theta}'] = clabels
    #
    ## save config
    #outputdir = os.path.join(
    #    FLAGS.output_basedir, 
    #    '{}/nclusters-{}/{}'.format(str(FLAGS.expname),str(FLAGS.nclusters), FLAGS.clf_key ))
    #os.makedirs(outputdir, exist_ok=True)
    #
    ## scores: dump to pickle
    #with open(os.path.join(outputdir, f'score-hac_{FLAGS.cexpname}.pkl'), 'wb') as f:
    #    pickle.dump(all_scores, f )
    #print("NORMAL END : SCORES")
    #
    ### replicate&rotate models: dump to pickle
    #with open(os.path.join(outputdir, f'rotlabel-hac_{FLAGS.cexpname}.pkl'), 'wb') as f:
    #    pickle.dump(all_labels, f )
    #print("NORMAL END : RCLUSTERING")
    
      
    ####  label
    olabel, oclustering = compute_hac(encoder,clf, patches)
    ### save 
    outputdir = os.path.join(
        FLAGS.output_basedir, 
        '{}/nclusters-{}/{}'.format(str(FLAGS.expname),str(FLAGS.nclusters), FLAGS.clf_key ))
    os.makedirs(outputdir, exist_ok=True)
    # original models: dump to pickle
    with open(os.path.join(outputdir, f'original-hac_{FLAGS.cexpname}.pkl'), 'wb') as f:
        pickle.dump(oclustering, f )
    print("NORMAL END : CLUSTERING")


    ## index selection
    #sindex_list = []
    #for cluster in range(FLAGS.nclusters):
    #  # first approach: extract first index
    #  sindex_list.append(np.where(olabel == cluster)[0][0])
    #  # second approach: extract data at random 
    #  #
    #  # TODO
    #sindex = np.asarray(sindex_list)
    #np.save(os.path.join(outputdir, 'sindex'), sindex )

    ## label creation /large_hac3
    #label = []
    #for ilabel in range(FLAGS.nclusters):
    #  tmp = [ ilabel for j in range(FLAGS.copy_size)]
    #  label.extend(tmp)
    #label = np.asarray(label)

    # label creation /large_hac1
    label = []
    for ilabel in olabel:
      tmp = [ ilabel for j in range(FLAGS.copy_size)]
      label.extend(tmp)
    label = np.asarray(label)

    ##### code for /large_hac3
    #""" Make original label"""
    ### copy patches
    #plist = []
    #spatches = patches[sindex]
    #n = spatches.shape[0] // FLAGS.alpha
    #for  i in range(n):
    #  plist.append(
    #    copy_rot_fn(spatches[i*FLAGS.alpha:(i+1)*FLAGS.alpha], 
    #                height=FLAGS.height, width=FLAGS.width, ch=FLAGS.channel, copy_size=FLAGS.copy_size)
    #  )
    #if spatches.shape[0] - n *FLAGS.alpha > 0:
    #  plist.append(copy_rot_fn(spatches[n*FLAGS.alpha:], 
    #                height=FLAGS.height, width=FLAGS.width, ch=FLAGS.channel, copy_size=FLAGS.copy_size)
    #  )
    #if len(plist) > 1:
    #  rpatches = np.concatenate(plist, axis=0)
    #else:
    #  rpatches = plist[0]
    #del plist, patches, spatches
    #gc.collect()

    ### code for /large_hac2 and /large_hac5
    
    #""" Make original label"""
    ## copy patches
    plist = []
    n = patches.shape[0] // FLAGS.alpha
    for  i in range(n):
      plist.append(
        copy_rot_fn(patches[i*FLAGS.alpha:(i+1)*FLAGS.alpha], 
                    height=FLAGS.height, width=FLAGS.width, ch=FLAGS.channel, copy_size=FLAGS.copy_size)
      )
    if patches.shape[0] - n *FLAGS.alpha > 0:
      plist.append(copy_rot_fn(patches[n*FLAGS.alpha:], 
                    height=FLAGS.height, width=FLAGS.width, ch=FLAGS.channel, copy_size=FLAGS.copy_size)
      )
    rpatches = np.concatenate(plist, axis=0)
    del plist, patches
    gc.collect()
    ##
    ### compute model 
    #label, oclustering = compute_hac(encoder, clf, rpatches)
    ### SAVE
    #outputdir = os.path.join(
    #    FLAGS.output_basedir, 
    #    '{}/nclusters-{}/{}'.format(str(FLAGS.expname),str(FLAGS.nclusters), FLAGS.clf_key ))
    #os.makedirs(outputdir, exist_ok=True)
    ### original models: dump to pickle
    #with open(os.path.join(outputdir, f'original-hac_{FLAGS.cexpname}.pkl'), 'wb') as f:
    #    pickle.dump(oclustering, f )
    #print("NORMAL END : CLUSTERING")

    ## large_hac5
    # Rotation by a theta degree: Get relatively larger theta 
    # to give enough angle of rotation
    #theta = math.pi/180 * np.random.randint(15,345,1)[0]   
    #n = rpatches.shape[0] // FLAGS.alpha
    #for idx, i in enumerate(range(n)):
    #  radians = [ theta for j in range(FLAGS.alpha) ]
    #  #print(radians)
    #  if idx == 0:
    #    r2patches = rotate_fn(rpatches[i*FLAGS.alpha:(i+1)*FLAGS.alpha], angles=radians) 
    #    r2patches_np = tf.keras.backend.eval(r2patches)
    #  else:
    #    tmp = rotate_fn(rpatches[i*FLAGS.alpha:(i+1)*FLAGS.alpha], angles=radians) 
    #    tmp_np = tf.keras.backend.eval(r2patches)
    #    r2patches_np = np.concatenate([r2patches_np, tmp_np], axis=0)
    #
    #if rpatches.shape[0] - n *FLAGS.alpha > 0:
    #    leftover = rpatches.shape[0] - n *FLAGS.alpha 
    #    radians = [ theta for j in range(leftover) ]
    #    tmp = rotate_fn(rpatches[n*FLAGS.alpha:], angles=radians) 
    #    tmp_np = tf.keras.backend.eval(r2patches)
    #    r2patches_np = np.concatenate([r2patches_np, tmp_np], axis=0)
    #
    #del rpatches, r2patches
    #gc.collect()

    ## compute model 
    #clabels, rclustering = compute_hac(encoder, clf, r2patches_np)
    #del r2patches_np
    #gc.collect()


    # /large_hac2
    #""" Rotated used-patches label (no copy process)"""
    #n = rpatches.shape[0] // FLAGS.alpha
    #for idx, i in enumerate(range(n)):
    #  radians = [j*math.pi/180 for j in np.random.randint(0,359,FLAGS.alpha) ]
    #  if idx == 0:
    #    r2patches = rotate_fn(rpatches[i*FLAGS.alpha:(i+1)*FLAGS.alpha], angles=radians) 
    #    r2patches_np = tf.keras.backend.eval(r2patches)
    #  else:
    #    tmp = rotate_fn(rpatches[i*FLAGS.alpha:(i+1)*FLAGS.alpha], angles=radians) 
    #    tmp_np = tf.keras.backend.eval(r2patches)
    #    r2patches_np = np.concatenate([r2patches_np, tmp_np], axis=0)

    #if rpatches.shape[0] - n *FLAGS.alpha > 0:
    #    leftover = rpatches.shape[0] - n *FLAGS.alpha 
    #    radians = [j*math.pi/180 for j in np.random.randint(0,359,int(leftover)) ]
    #    tmp = rotate_fn(rpatches[i*FLAGS.alpha:(i+1)*FLAGS.alpha], angles=radians) 
    #    tmp_np = tf.keras.backend.eval(r2patches)
    #    r2patches_np = np.concatenate([r2patches_np, tmp_np], axis=0)
    #del rpatches, r2patches
    #gc.collect()
    # compute model 
    #clabels, rclustering = compute_hac(encoder, clf, r2patches_np)
    #del r2patches_np
    #gc.collect()

    # compute model 
    clabels, rclustering = compute_hac(encoder, clf, rpatches)
    del rpatches
    gc.collect()


    ## compute scores
    all_scores = {}
    scores = {}
    for ikey in scoring:
      if ikey == 'ami':
        tmp = AMI(label, clabels)
      elif ikey == 'nmi':
        tmp = NMI(label, clabels)
      elif ikey == 'cls':
        tmp = CLS(label, clabels)
      elif ikey == 'hs':
        tmp = HS(label, clabels)
      elif ikey == 'ars':
        tmp = ARS(label, clabels)
      scores[ikey] = tmp
    ## store as dict
    all_scores[f'nclusters-{FLAGS.nclusters}'] = scores

    ### save 
    outputdir = os.path.join(
        FLAGS.output_basedir, 
        '{}/nclusters-{}/{}'.format(str(FLAGS.expname),str(FLAGS.nclusters), FLAGS.clf_key ))
    os.makedirs(outputdir, exist_ok=True)

    ## scores: dump to pickle
    with open(os.path.join(outputdir, f'score-hac_{FLAGS.cexpname}.pkl'), 'wb') as f:
        pickle.dump(all_scores, f )
    print("NORMAL END : SCORES")

    ## replicate&rotate models: dump to pickle
    with open(os.path.join(outputdir, f'reprot-hac_{FLAGS.cexpname}.pkl'), 'wb') as f:
        pickle.dump(rclustering, f )
    print("NORMAL END : RCLUSTERING")
    
       
