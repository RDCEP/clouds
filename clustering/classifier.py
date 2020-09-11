# Classification for K-fold cross validation
# Coded by Takuya Kurihana Sep 09 2020
#
author = "tkurihana@uchicago.edu"

import os
import gc
import sys
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
#from multiprocessing import Pool
from mpi4py   import MPI

#version1
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import metrics
from sklearn.neural_network import MLPClassifier 
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#version 2
from sklearn.model_selection import cross_validate
#from sklearn.metrics import recall_score
#from sklearn.model_selection import KFold
#from sklearn.model_selection import RepeatedKFold
#from sklearn.model_selection import ShuffleSplit


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
    return rtest_imgs

def read_txt():
    # save the pair patch information
    with open('./pair_index.txt', 'r') as f:
      data = []
      lines = f.read().split('\n')
      for line in lines:
        if len(line) > 0:
            data.append(int(line))
    return data


def read_txt2():
    with open('./pair_index2.txt', 'r') as f:
      data1 = []
      data2 = []
      lines = f.read().split('\n')
      for line in lines:
        tmp = line.split(',')
        #print(tmp)
        if len(tmp) > 1:
            data1.append(int(tmp[0]))
            data2.append(int(tmp[1]))
    return data1, data2

def read_txt3():
    with open('./pair_index6_ri.txt', 'r') as f:
    #with open('./pair_index6_nri.txt', 'r') as f:
    #with open('./pair_index3.txt', 'r') as f:
    #with open('./pair_index5.txt', 'r') as f:
      data1 = []
      data2 = []
      lines = f.read().split('\n')
      for line in lines:
        tmp = line.split(',')
        #print(tmp)
        if len(tmp) > 1:
            data1.append(int(tmp[0]))
            data2.append(int(tmp[1]))
    return data1, data2

def read_txt4(txtname):
    with open('./'+txtname, 'r') as f:
      data1 = []
      data2 = []
      lines = f.read().split('\n')
      for line in lines:
        tmp = line.split(',')
        #print(tmp)
        if len(tmp) > 1:
            data1.append(int(tmp[0]))
            data2.append(int(tmp[1]))
    return data1, data2


def run(args):
  return _run(*args)


def _run(model_datadir, expname, patches, idx1, idx2, height, width, channel, copy_size,cv,clf_key):
    # classifier load model
    model_dir = os.path.join(model_datadir,str(expname) )
    encoder = load_latest_model(model_dir, mtype='encoder')

    # score metrics
    scoring = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'average_precision',
        'roc_auc', 
    ]

    # model metrics
    clf_dict = {
      'SVM': svm.SVC(kernel='linear', C=1, random_state=0),
      'MLP': MLPClassifier(alpha=1, max_iter=5000),
      'RF': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
      'ADABOOST':AdaBoostClassifier(),

    }
    # label
    label = np.array([0 for i in range(FLAGS.copy_size)]+[1 for j in range(FLAGS.copy_size)])

    scores = {}
    clf = clf_dict[clf_key]
    for (_idx1, _idx2) in zip(idx1, idx2):
      print(_idx1, _idx2, flush=True)
      # select two patches and make them a pair 
      spatches = make_pairdataset(patches, _idx1, _idx2)
      print('make spatches', flush=True)

      # copy patches
      rspatches = copy_rot_fn(spatches, 
                  height=height, width=width, ch=channel, copy_size=copy_size)
      print('make rspatches', flush=True)


      # compute model 
      print('start cross validation', flush=True)
      encs = encoder.predict(rspatches)
      n,h,w,c = encs.shape
      _scores = cross_validate(clf, encs.reshape(n, h*w*c), label, cv=cv, scoring=scoring)
      scores[_idx1] = _scores
      print('end cross validation', flush=True)

    return scores


def get_argument(verbose=True):
    p = argparse.ArgumentParser()
    p.add_argument('--tf_datadir', type=str, default=None)
    p.add_argument('--output_basedir', type=str, default=None)
    p.add_argument('--model_datadir', type=str, default=None)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--channel', type=int, default=None)
    p.add_argument('--expname', type=str, default=None)
    p.add_argument('--cexpname', type=str, default=None)
    p.add_argument('--copy_size', type=int, default=None)
    p.add_argument('--cv', type=int, default=None)
    p.add_argument('--clf_key', type=str, default=None)
    p.add_argument('--nproc', type=int, default=None)
    args = p.parse_args()
    if verbose:
      for f in args.__dict__:
        print("\t", f, (25 - len(f)) * " ", args.__dict__[f])
      print("\n")
    return args

if __name__ == "__main__":
    #npatches = 99 # number of patches - 1 
    npatches = 19 # number of patches - 1 
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # create argument parser
    FLAGS = get_argument(verbose=rank == 0)

    # get patches from holdlout tfrecord
    patches = load_dataset(datadir=FLAGS.tf_datadir, filebasename='2-10*.tfrecord', 
                height=FLAGS.height, width=FLAGS.width,channel=FLAGS.channel)


    """ Pair.txt version
    """
    ## read pair index text file
    #idx2_list = read_txt()
    #idx1s = []
    #idx2s = []
    ##for idx, (_idx1, _idx2) in enumerate(zip(range(len(idx2_list)), idx2_list )):
    #x = np.array([i for i in range(len(idx2_list))])
    ##random.shuffle(x)
    #idx1_list = [759] + [ i for i in x[:npatches]]
    #for idx, _idx1 in enumerate(idx1_list):
    #    if idx % size == rank:
    #        idx1s.append(_idx1)
    #        _idx2 = idx2_list[_idx1]
    #        idx2s.append(_idx2)
    """ Pair.txt for all patches version 
    """
    #idx1s = []
    #idx2s = []
    #idx1_list, idx2_list = read_txt3()
    #for idx, (idx1, idx2) in enumerate(zip(idx1_list, idx2_list)):
    #    if idx % size == rank:
    #        idx1s.append(idx1)
    #        idx2s.append(idx2)

    """ Pair2.txt version
    """
    #idx1s = []
    #idx2s = []
    #_idx1_list, _idx2_list = read_txt2()
    #idx1_list = [759]   + [ i for i in _idx1_list[:npatches]]
    #idx2_list = [1926]  + [ i for i in _idx2_list[:npatches]]
    #for idx, (idx1, idx2) in enumerate(zip(idx1_list, idx2_list)):
    #    if idx % size == rank:
    #        idx1s.append(idx1)
    #        idx2s.append(idx2)
    
    """ Pair3 and 5. 6. txt version
    """
    idx1s = []
    idx2s = []
    idx1_list, idx2_list = read_txt3()
    for idx, (idx1, idx2) in enumerate(zip(idx1_list, idx2_list)):
        if idx % size == rank:
            idx1s.append(idx1)
            idx2s.append(idx2)

    """ Pairr.txt version
    """
    #idx1s = []
    #idx2s = []
    #if str(FLAGS.expname) == "67011582" :
    #    txtname = "pair_index4_ri.txt"
    #elif str(FLAGS.expname) == "m2_02_global_2000_2018_band28_29_31" :
    #    txtname = "pair_index4_nri.txt"
    #
    #idx1_list, idx2_list = read_txt4(txtname)
    #for idx, (idx1, idx2) in enumerate(zip(idx1_list, idx2_list)):
    #    if idx % size == rank:
    #        idx1s.append(idx1)
    #        idx2s.append(idx2)
    #
    #
    if not idx1s:
        raise ValueError("Error in pair generation")


    #stime = time.time()  # initial time  
    # multiprocess
    #procs = Pool(FLAGS.nproc)
    # encoder, patches, idx1, idx2, height, width, channel, copy_size,cv,clf_key
    #args = [ ( FLAGS.model_datadir,FLAGS.expname, patches, idx1, idx2, 
    #          FLAGS.height, FLAGS.width, FLAGS.channel, 
    #          FLAGS.copy_size,FLAGS.cv,FLAGS.clf_key 
    #         )  for (idx1, idx2) in zip(
    #          np.array_split(np.arange(0,len(idx2_list),1), FLAGS.nproc), np.array_split(np.array(idx2_list),FLAGS.nproc)) ]
    #all_scores = procs.map(run, args)
    #_all_scores = procs.map(run, args)
    
    #print(_all_scores)
    #all_scores  = {}

    #etime = (time.time() -stime)/60.0 # minutes
    #print("   Execution time [minutes]  : %f" % etime, flush=True)


    """ Original """
    # classifier load model
    model_dir = os.path.join(FLAGS.model_datadir,str(FLAGS.expname) )
    encoder = load_latest_model(model_dir, mtype='encoder')

    # score metrics
    scoring = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'average_precision',
        'roc_auc', 
    ]

    # model metrics
    clf_dict = {
      'SVM': svm.SVC(kernel='linear', C=1, random_state=0),
      'MLP': MLPClassifier(alpha=1, max_iter=5000),
      'RF': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
      'ADABOOST':AdaBoostClassifier(),

    }
    # label
    label = np.array([0 for i in range(FLAGS.copy_size)]+[1 for j in range(FLAGS.copy_size)])
    all_scores = {}
    #for idx, (idx1, idx2) in enumerate(zip(range(len(idx2_list)), idx2_list )):
    for idx, (idx1, idx2) in enumerate(zip(idx1s, idx2s)):
        if rank == 0:
          print(idx1, idx2, flush=True)
        # select two patches and make them a pair 
        spatches = make_pairdataset(patches, idx1,idx2)

        # copy patches
        rspatches = copy_rot_fn(spatches, 
                      height=FLAGS.height, width=FLAGS.width, ch=FLAGS.channel, copy_size=FLAGS.copy_size)


        # compute model 
        clf = clf_dict[FLAGS.clf_key]
        encs = encoder.predict(rspatches)
        gc.collect()
        n,h,w,c = encs.shape
        scores = cross_validate(clf, encs.reshape(n, h*w*c), label, cv=FLAGS.cv, scoring=scoring)

        # store as dict
        all_scores[f'idx-{idx1}-{idx2}'] = scores

        del rspatches, spatches

    # save
    outputdir = os.path.join(
        FLAGS.output_basedir, 
        '{}/cv-{}/{}'.format(str(FLAGS.expname),str(FLAGS.cv), FLAGS.clf_key ))
    os.makedirs(outputdir, exist_ok=True)

    # dump to json
    #json.dump(all_scores, open(os.path.join(outputdir, f'scores_{FLAGS.cexpname}-{rank}.json'), 'w'))    
    with open(os.path.join(outputdir, f'scores_{FLAGS.cexpname}-{rank}.pkl'), 'wb') as f:
        pickle.dump(all_scores, f )
    print("NORMAL END at rank {}".format(rank))
        

