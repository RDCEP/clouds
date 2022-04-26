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
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.data.experimental import parallel_interleave
from tensorflow.python.keras.models import Model

#version1
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering as HAC
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARS
from sklearn.metrics import completeness_score as CLS
from sklearn.metrics import  homogeneity_score as HS
from sklearn.neighbors import NearestCentroid

# customized code
#from hierarchical import AgglomerativeClustering as HAC

#joblib for memry
#from joblib import Memory
"""memorystr or object with the joblib.Memory interface, default=None
Used to cache the output of the computation of the tree. 
By default, no caching is done. 
If a string is given, it is the path to the caching directory.

example of joblib.Memory
https://qiita.com/Tatejimaru137/items/c3aabd17196543fdfd20

joblib documentation
https://joblib.readthedocs.io/en/latest/generated/joblib.Memory.html

"""


tf.compat.v1.disable_eager_execution()
def data_extractor_resize_fn(filelist,prefetch=1,height=32,width=32,read_threads=1, distribute=(1, 0), resize_flag=True):
    def parser(ser):
        """
        read_threads should be 1 o.w. order of patches will mess up and results in incorrect analysis reuslts
        when you want to identify number of clusters tied to each patch in clustering training data

        Decode & Pass datast in tf.record
        *Cuation*
        floating point: tfrecord data ==> tf.float64
        """
        features = {
            "shape": tf.io.FixedLenFeature([3], tf.int64),
            "patch": tf.io.FixedLenFeature([], tf.string),
            "filename": tf.io.FixedLenFeature([], tf.string),
            "coordinate": tf.io.FixedLenFeature([2], tf.int64),
        }
        decoded = tf.io.parse_single_example(ser, features)
        patch = tf.reshape(
            tf.io.decode_raw(decoded["patch"], tf.float64), decoded["shape"]
        )
        # conversion of tensor
        patch = tf.cast(patch, tf.float32)
        return patch
    
    dataset = (
        tf.data.Dataset.list_files(filelist, shuffle=False)
            .shard(*distribute)
            .apply(
            parallel_interleave(
                lambda f: tf.data.TFRecordDataset(f).map(parser),
                cycle_length=read_threads,
                sloppy=True,
            )
        )
    )
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    next_element = iterator.get_next()
    idx = 0
    patches = None
    with tf.compat.v1.Session() as sess:
        try:
            while True:
                patch = sess.run(next_element)
                if idx == 0:
                    patches = np.expand_dims(patch, axis=0)
                else:
                    patches = np.concatenate(
                               [patches, np.expand_dims(patch, axis=0)],axis=0
                              )
                idx+=1
        except tf.errors.OutOfRangeError:
            print("OutOfRage --> finish process",flush=True)
            pass
    return patches

def load_latest_model(model_dir, mtype):
    latest = 0, None
    # get trained wegiht 
    for m in os.listdir(model_dir):
        if ".h5" in m and mtype in m:
            epoch = int(m.split("-")[1].replace(".h5", ""))
            latest = max(latest, (epoch, m))

    epoch, model_file = latest

    if not os.listdir(model_dir):
        raise NameError("no directory. check model path again")

    print(" Load {} at {} epoch".format(mtype, epoch),flush=True)
    model_def = model_dir+'/'+mtype+'.json'
    model_weight = model_dir+'/'+mtype+'-'+str(epoch)+'.h5'
    with open(model_def, "r") as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(model_weight)
    return model


def load_dataset(datadir=None, ext='.tfrecord', height=None, width=None,channel=None, 
                 resize_flag=True,date=None,layer_name=None):

    # get files
    _filelist = glob.glob(os.path.join(datadir, f"*{ext}" ))
    # sort them
    filelist = sorted(_filelist)
    # save order of tfrecord filenames
    np.save(f"./sort_file_names_{layer_name}-{date}.npy", filelist)

    ## load 
    patches_list = []
    for ifile in filelist:
      patches_list.append(data_extractor_resize_fn([ifile],height=height,width=width,read_threads=1, resize_flag=resize_flag))
    print("NORMAL END",flush=True)

    ## get patch
    patches = np.concatenate(patches_list,axis=0)
    print("PATCH SHAPE", patches.shape,flush=True)
    return patches

def compute_hac(encoder, decoder=None, clf=None,patches=None,layer_name="leaky_re_lu_23", search_all=False):
    """ 

    OUT
      label
      clustring model
      latent representation
    """
    results = {}
    ### use bottleneck layer
    encs = encoder.predict(patches)
    del patches
    gc.collect()

    #n,h,w,c = encs.shape
    #clustering = clf.fit(encs.reshape(n, h*w*c))
    #clabels = clustering.labels_
    #results["encoder-labels"] = clabels
    #results["encoder-rep"] = encs
    print('END Encoder', flush=True)

    ### use a layer from the decoder of RI model
    # decoder
    if search_all:
      model=decoder
      layer_names = [layer.name for layer in model.layers]
      for layer_name in layer_names[1:]:
        if  're_lu' in layer_name:
          rep  = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
          pred = rep.predict(encs)
      
          n,h,w,c = pred.shape
          clustering = clf.fit(pred.reshape(n, h*w*c))
          clabels = clustering.labels_
          results[f"{layer_name}-labels"] = clabels
          #results[f"{layer_name}-clfs"] = clustering
          # off for stability test
          #results[f"{layer_name}-rep"] = pred
          print(f"END Decoder at {layer_name}", flush=True)
          gc.collect()

      return results

    else:
        # get one specified layer from Decoder
        model = decoder
        rep  = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        pred = rep.predict(encs)
        print("END PREDICTION", flush=True)

        n,h,w,c = pred.shape
        clustering = clf.fit(pred.reshape(n, h*w*c))
        clabels = clustering.labels_
        results[f"{layer_name}-labels"] = clabels
        #results[f"{layer_name}-clfs"] = clustering
        #results[f"{layer_name}-rep"] = pred
        # return clabels, clustering, pred
        print(f'END LAYER {layer_name} : save latent rep', flush=True)
        #print(f'END LAYER {layer_name} : no save latent rep', flush=True)

        # compute centroids
        centroids = comp_centroids(rep=pred, labels=clabels)

        del pred
        return results, centroids

def compute_hac_bottle(encoder, decoder=None, clf=None,patches=None,layer_name="leaky_re_lu_23", search_all=False):
    """ 

    OUT
      label
      clustring model
      latent representation
    """
    results = {}
    ### use bottleneck layer
    encs = encoder.predict(patches)
    del patches
    gc.collect()

    n,h,w,c = encs.shape
    clustering = clf.fit(encs.reshape(n, h*w*c))
    clabels = clustering.labels_
    results["encoder-labels"] = clabels
    results["encoder-clfs"] = clustering
    results["encoder-rep"] = encs
    print('END Encoder', flush=True)
    return results


def comp_centroids(rep, labels):
    # shape
    n,h,w,c = rep.shape
    _rep = rep.reshape(n,h*w*c)
    # centroids
    clf = NearestCentroid()
    clf.fit(_rep, labels)
    centroids = clf.centroids_
    return centroids


def eval_fn(labels=None, clabels=None, scoring=None):
    #all_scores = {}
    scores = {}
    for ikey in scoring:
      if ikey == 'ami':
        tmp = AMI(labels, clabels)
      elif ikey == 'nmi':
        tmp = NMI(labels, clabels)
      elif ikey == 'cls':
        tmp = CLS(labels, clabels)
      elif ikey == 'hs':
        tmp = HS(labels, clabels)
      elif ikey == 'ars':
        tmp = ARS(labels, clabels)
      scores[ikey] = tmp
    # store as dict
    return scores


def get_argument(verbose=True):
    p = argparse.ArgumentParser()
    subparsers = p.add_subparsers(help='Arguments for specific clustering types.', dest='clustering_type')
    subparsers.required = True


    hac_parser = subparsers.add_parser('HAC')
    #hac_parser.add_argument('--full_tree', action='store_true')


    db_parser = subparsers.add_parser('DBSCAN')
    db_parser.add_argument('--eps', type=float)
    db_parser.add_argument('--min_samples', type=int)
    db_parser.add_argument('--leaf_size',   type=int)
    db_parser.add_argument('--n_jobs',      type=int)


    # common param
    p.add_argument('--tf_holdout_datadir', type=str, default=None)
    p.add_argument('--tf_subset_datadir', type=str, default=None)
    p.add_argument('--output_basedir', type=str, default=None)
    p.add_argument('--model_datadir', type=str, default=None)
    p.add_argument('--cache_datadir', type=str, default=None)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--channel', type=int, default=None)
    p.add_argument('--expname', type=str, default=None)
    p.add_argument('--date', type=str, default=None)
    p.add_argument('--clf_key', type=str, default=None)
    p.add_argument('--layer_name', type=str, default=None)
    p.add_argument('--resize_flag', action='store_true', help='resize input tfrecord')
    p.add_argument('--search_all', action='store_true', help='search all leasky ReLU layer in decoder')
    # list 
    p.add_argument("--nclusters",nargs='+',
        help="List of number of cluser we test  ",
    )
    p.add_argument('--full_tree', action='store_true')

    # integer
    #p.add_argument('--nclusters', type=int, default=None)
    args = p.parse_args()
    if verbose:
      for f in args.__dict__:
        print("\t", f, (25 - len(f)) * " ", args.__dict__[f], flush=True)
      print("\n",flush=True)
    return args

if __name__ == "__main__":

    # create argument parser
    FLAGS = get_argument(verbose=True)

    stime = time.time()
    """ Modify here to leave out patches with less gradation """
    # holdout
    Hpatches = load_dataset(datadir=FLAGS.tf_holdout_datadir, height=FLAGS.height, width=FLAGS.width,
                           channel=FLAGS.channel, resize_flag=FLAGS.resize_flag, date=f"H-{FLAGS.date}",
                           layer_name=FLAGS.layer_name)
    # subset
    Spatches = load_dataset(datadir=FLAGS.tf_subset_datadir, height=FLAGS.height, width=FLAGS.width,
                           channel=FLAGS.channel, resize_flag=FLAGS.resize_flag, date=FLAGS.date,
                           layer_name=FLAGS.layer_name)
    # Show collected data 
    print(f" Number of  holdout of patches {Hpatches.shape[0]}", flush=True)
    print(f" Number of  random subset of patches {Spatches.shape[0]}", flush=True)

    # merge H and S
    # clip patches if H > 14000 and/or S > 56000
    if Hpatches.shape[0] > 14000:
        Hpatches = Hpatches[:14000]
    if Spatches.shape[0] > 56000:
        Spatches = Spatches[:56000]
    patches = np.concatenate([Hpatches, Spatches], axis=0)

    print(f" Number of patches {patches.shape[0]}", flush=True)
    etime = time.time() - stime
    print(f"PIPELINE EXECUTION TIME {etime/60} MIN", flush=True)

    # classifier load model
    model_dir = os.path.join(FLAGS.model_datadir,str(FLAGS.expname) )
    encoder = load_latest_model(model_dir, mtype='encoder')
    decoder = load_latest_model(model_dir, mtype='decoder')
    gc.collect()

    for  icluster in FLAGS.nclusters:
      ncluster = int(icluster)  # need to be interger
      print(f"START CLUSTER {ncluster}", flush=True)

      # model metrics
      os.makedirs(FLAGS.cache_datadir, exist_ok=True)

      # add joblib.Memory
      clf_dict = {
        'HAC': HAC(n_clusters=ncluster,memory=FLAGS.cache_datadir,
                  compute_full_tree=FLAGS.full_tree,compute_distances=True) ,
      }
        # if you want to use dbscan comment off below and add
        #'DBSCAN': DBSCAN(eps=FLAGS.eps, min_samples=FLAGS.min_samples, metric='euclidean', 
        #                 metric_params=None, algorithm='auto', 
        #                 leaf_size=FLAGS.leaf_size, p=None, n_jobs=FLAGS.n_jobs)
      # define a model selected from metrics
      clf = clf_dict[FLAGS.clf_key]

      # compute model 
      stime = time.time()
      print("START CLUSTERING",flush=True)
      results, centroids = compute_hac(encoder, decoder, clf=clf, patches=patches,
                          layer_name=FLAGS.layer_name, search_all=FLAGS.search_all)
      ### For RA and NRI autoencoder
      #results = compute_hac_bottle(encoder,  clf=clf, patches=patches,
      #                      search_all=FLAGS.search_all)
      gc.collect()

      ### save 
      if FLAGS.search_all:
        outputdir = os.path.join(
          FLAGS.output_basedir, 
          '{}/nclusters-{}/{}/{}'.format(str(FLAGS.expname),str(ncluster), FLAGS.clf_key, FLAGS.date ))
        os.makedirs(outputdir, exist_ok=True)
        #  '{}/nclusters-{}/{}'.format(str(FLAGS.expname),str(FLAGS.nclusters), FLAGS.clf_key ))
      else:
        outputdir = os.path.join(
          FLAGS.output_basedir, 
          '{}/nclusters-{}/{}/{}/{}'.format(str(FLAGS.expname),str(ncluster), FLAGS.clf_key, FLAGS.layer_name, FLAGS.date ))
        os.makedirs(outputdir, exist_ok=True)
        #  '{}/nclusters-{}/{}/{}'.format(str(FLAGS.expname),str(FLAGS.nclusters), FLAGS.clf_key, FLAGS.layer_name ))

      ## scores: dump to pickle
      with open(os.path.join(outputdir, f'score-hac_{FLAGS.expname}_{FLAGS.date}-data.pkl'), 'wb') as f:
      #with open(os.path.join(outputdir, f'score-hac_{FLAGS.expname}_2003-data.pkl'), 'wb') as f:
      #with open(os.path.join(outputdir, f'score-{FLAGS.clf_key}_{FLAGS.expname}_{FLAGS.eps}_{FLAGS.min_samples}.pkl'), 'wb') as f:
          pickle.dump(results, f )

      ## centroids
      outputfilename = f"hac_ncluster{ncluster}-centroids-{FLAGS.expname}.npy"
      cfilename = os.path.join(outputdir, outputfilename)
      np.save(cfilename, centroids)

      print(f"NORMAL END : {FLAGS.expname}")
      etime = time.time() - stime
      print(f"EXECUTION TIME {etime/60} MIN", flush=True)
    
    
       
