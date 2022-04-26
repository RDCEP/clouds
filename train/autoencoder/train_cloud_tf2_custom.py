__author__ = "tkurihana@uchicago.edu"

import os
import sys
import glob
import json
import time
import math
import argparse
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from tensorflow.image import image_gradients
from tensorflow.python.client import timeline
from tensorflow.data.experimental import parallel_interleave

# horovod
from horovod import tensorflow as hvd

# version check
print(f"tensorflow == {tf.__version__}", flush=True)


src_dir='./'
sys.path.insert(1,os.path.join(sys.path[0],src_dir))

from models_resnet import model_resnet_fn
from models_update import model_synmetric_resize_fn

def get_args(verbose=False):
  p = argparse.ArgumentParser()
  p.add_argument(
    '--logdir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--figdir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--input_datadir',
    nargs="+",
    default='list of clouds tfrecord data directories'
  )
  p.add_argument(
    '--output_modeldir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--lr',
    type=float,
    default=0.001
  )
  p.add_argument(
    '--expname',
    type=str,
    default='new'
  )
  p.add_argument(
    '--num_epoch',
    type=int,
    default=5
  )
  p.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='number of pictures in minibatch'
  )
  p.add_argument(
    '--npatches',
    type=int,
    default=2000,
    help='number of patches/tfrecord'
  )
  p.add_argument(
    '--height',
    type=int,
    default=32
  )
  p.add_argument(
    '--width',
    type=int,
    default=32
  )
  p.add_argument(
    '--channel',
    type=int,
    default=1
  )
  p.add_argument(
    '--nblocks',
    type=int,
    default=5
  )
  p.add_argument(
    '--base_dim',
    type=int,
    help="coef for size of filter at first convolutional layer in first block in encoder. 2**(base_dim)",
    default=4
  )
  p.add_argument(
    '--nstack_layer',
    type=int,
    help="Number of convolution layers/block",
    default=3
  )
  p.add_argument(
      '--retrain',
      action="store_true",
      help='attach this FLAGS if you need retraining from trained code',
      default=False
  )
  p.add_argument(
      '--retrain_datadir',
      type=str,
      help='you should express directory if you retrain from a trained model',
      default='./'
  )
  p.add_argument(
    '--save_every',
    type=int,
    default=10
  )
  p.add_argument(
    '--copy_size',
    type=int,
    default=32
  )
  p.add_argument(
    '--f_lambda',
    type=float,
    help='lambda coef for transform invariant term',
    default=1.0
  )
  p.add_argument(
    '--c_lambda',
    type=float,
    help='lambda coef for restoration term',
    default=1.0
  )
  p.add_argument(
    '--s_lambda',
    type=float,
    help='lambda coef for sparse term',
    default=1.0
  )
  p.add_argument(
    '--degree',
    type=int,
    help='how much degree you rotate in loss term',
    default=1
  )
  p.add_argument(
      '--resnet',
      action="store_true",
      help='attach this FLAGS if you want to train resnet model; then nblocks -1 ',
      default=False
  )
  # SHOW config on outfile
  args = p.parse_args()
  if verbose:
    for f in args.__dict__:
      print("\t", f, (25 - len(f)) * " ", args.__dict__[f])
    print("\n")
  return args


def rotate_fn(images, angles):
    """
    TF2
    images : 4d tensor [batch, height, width channel]
      original oprion - nearest. Additional: biilnear
    """
    # v2.x
    rotated_tensor_images = tfa.image.transform(
      images,
      tfa.image.transform_ops.angles_to_projective_transforms(
        angles, tf.cast(tf.shape(images)[1], tf.float32), 
            tf.cast(tf.shape(images)[2], tf.float32)
        ),
    )
    return rotated_tensor_images

def resize_image_fn(imgs,height=32, width=32):
  """Resize images from (#batch,H',W',C) --> (#batch,H,W,C)
  """
  reimgs = tf.image.resize(imgs, (height, width))
  return reimgs


def input_clouds_fn(filelist, height=32, width=32, batch_size=32, copy_size=4, prefetch=1, read_threads=4, distribute=(1, 0)):
    """
      INPUT:
        prefetch: tf.int64. How many "minibatch" we asynchronously prepare on CPU ahead of GPU
    """

    def parser(ser):
        """
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
        oheight = decoded["shape"][0]
        # conversion of tensor
        patch = tf.cast(patch, tf.float32)
        return patch

    # check batch/copy ratio
    try:
        if hvd.rank() == 0:
          if batch_size % copy_size == 0:
            print("\n Number of actual original images == {} ".format(int(batch_size)))
    except:
        raise ValueError("\n Division of batch size and copy size is not Integer \n")

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
    dataset = dataset.shuffle(1000).cache().repeat().batch(batch_size).prefetch(prefetch)
    return dataset


def mapper_fn(oimgs_tf,height=32, width=32, channel=1):
  """
      output should be two tf.Tensor o.w. No gradients provided for any variable
  """
  return oimgs_tf, oimgs_tf

def load_latest_model_weights(model, model_dir, name):
    """
      INPUT:
        model: encoder or decoder
        model_dir: model directory 
        name: model name.

      OUTPUT:
        step: global step 
    """
    latest = 0, None
    # get trained wegiht 
    for m in os.listdir(model_dir):
        if ".h5" in m and name in m:
            step = int(m.split("-")[1].replace(".h5", ""))
            latest = max(latest, (step, m))

    step, model_file = latest

    if not os.listdir(model_dir):
        raise NameError("no directory. check model path again")

    if model_file:
        model_file = os.path.join(model_dir, model_file)
        model.load_weights(model_file)
        print(" ... loaded weights for %s from %s", name, model_file)

    else:
        print("no weights for %s in %s", name, model_dir)

    return step

def loss_l2(imgs, encoder, decoder):
  encoded_imgs = encoder(imgs, training=True)
  decoded_imgs = decoder(encoded_imgs, training=True)
  return tf.reduce_mean(tf.square(decoded_imgs - imgs))

def loss_sparse_fn(img, encoder, batch_size=32):
    """
      Spatial sparseness term for const function
      Sum_{i \in Set of imgs in minibatch} [L1(E(img)) / L2(E(img))]^2
    """
    encoded_imgs = encoder(img, training=True)
    loss_sparse = 0.00
    for i in range(batch_size):
      L1 = tf.norm(encoded_imgs[i], ord=1)
      L2 = tf.norm(encoded_imgs[i], ord=2)
      loss_sparse += tf.square(L1/L2)
    loss_sparse /= batch_size # alternative to reduce mean 
    return loss_sparse


def model_map_fn(model, imgs):
    return tf.map_fn(model, imgs)

def transform_invariant_fn(imgs,
                           encoder,
                           decoder,
                           batch_size=32,
                           degree=1):
    """ Combine three loss functions into this function
      Major update
      - Combine nesting into one loop
      - Apply tf.map_fn for encoder/decoder operation
     """
  
    encoded_imgs = encoder(imgs, training=True)
    decoded_imgs = decoder(encoded_imgs, training=True)

    # first case
    angle_tf = tf.constant([ degree*math.pi/180.00 for i in range(batch_size)], dtype=tf.float32)
    rimgs = rotate_fn(imgs,angles=angle_tf) # R_theta(x_hat)
    rencoded_imgs = encoder(rimgs, training=True)
    drimgs = decoder(rencoded_imgs, training=True)
    loss_transform = tf.square(decoded_imgs - drimgs)
    loss_rest = tf.expand_dims(
        tf.reduce_mean(tf.square(rimgs - decoded_imgs), axis=[1,2,3]), axis=0
    )


    degree_tf = tf.constant(degree)
    start = tf.constant(degree*2)
    limit = tf.constant(360)

    for x in range(int(degree*2), 360, degree):
      _x = tf.cast(x, dtype=tf.float32)
      angle = _x * math.pi/180.00 # tensor
      angle_tf = tf.fill([batch_size], angle)
      rimgs = rotate_fn(imgs,angles=angle_tf) # R_theta(x_hat)
      rencoded_imgs = encoder(rimgs, training=True)
      drimgs = decoder(rencoded_imgs, training=True)
     
      # paper version
      loss_transform += tf.square(decoded_imgs - drimgs)
      tmp = tf.expand_dims(
          tf.reduce_mean(tf.square(rimgs - decoded_imgs), axis=[1,2,3]), axis=0
      )
      loss_rest = tf.concat([loss_rest,tmp],axis=0)

    # Average over degree
    loss_transform = loss_transform / (360.00/degree)
    loss_transform = tf.reduce_mean(loss_transform)
    loss_rest_tf = tf.reduce_mean(tf.reduce_min(loss_rest, axis=0)) 

    loss_sparse = 0.00
    for i in range(batch_size):
      L1 = tf.norm(encoded_imgs[i], ord=1)
      L2 = tf.norm(encoded_imgs[i], ord=2)
      loss_sparse += tf.square(L1/L2)
    loss_sparse /= batch_size # alternative to reduce mean 
    return loss_transform, loss_rest_tf, loss_sparse


### Loss fron NRI autoencoder Kurihana et al. 2019
def loss_fn(name, weight, fn, **kwargs):
    """Helper fn to add summary, scope, and nonzero weight condition.
    """
    if weight:
       loss = fn(**kwargs)
       return loss * weight
    return 0

def _image_losses(img, ae_img, w_mse=1, w_mae=1, w_hfe=1, w_ssim=1):
#def _image_losses(img, encoder, decoder, w_mse=1, w_mae=1, w_hfe=1, w_ssim=1):
    """Applies the 4 image losses.
    - Mean Square Error (L2 Loss)
    - Mean Absolute Error
    - High Frequency Error (mean abs difference after passing edge detectors)
    - Multi-scale structural similarity (ensure similar image mean/stdev/correlation)
    """

    def hfe():
        dx_img, dy_img = image_gradients(img)
        dx_ae_img, dy_ae_img = image_gradients(ae_img)
        return tf.reduce_mean(tf.abs(dx_img - dx_ae_img) + tf.abs(dy_img - dy_ae_img))

    def msssim():
        s = tf.image.ssim_multiscale(img, ae_img, max_val=5, power_factors=[1, 1, 1])
        return 1 - tf.reduce_mean(s)

    mse = lambda: tf.reduce_mean((img - ae_img) ** 2)
    mae = lambda: tf.reduce_mean(tf.abs(img - ae_img))

    l = 0
    l += loss_fn("mean_square_error", w_mse, mse)
    l += loss_fn("mean_abs_error", w_mae, mae)
    l += loss_fn("high_frequency_error", w_hfe, hfe)
    l += loss_fn("ms_ssim", w_ssim, msssim)
    return l


def load_latest_model_weights(model, model_dir, name):
    """
      INPUT:
        model: encoder or decoder
        model_dir: model directory 
        name: model name.

      OUTPUT:
        step: global step 
    """
    latest = 0, None
    # get trained wegiht 
    for m in os.listdir(model_dir):
        if ".h5" in m and name in m:
            step = int(m.split("-")[1].replace(".h5", ""))
            latest = max(latest, (step, m))

    step, model_file = latest

    if not os.listdir(model_dir):
        raise NameError("no directory. check model path again")

    if model_file:
        model_file = os.path.join(model_dir, model_file)
        model.load_weights(model_file)
        print(" ... loaded weights for %s from %s", name, model_file)

    else:
        print("no weights for %s in %s", name, model_dir)

    return step

if __name__ == '__main__':
  # init hvd
  hvd.init()

  ### GPU 
  # V.2: Pin GPU to be used to process local rank (one GPU per process)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
  if hvd.rank() == 0:
    print("Number of GPUs {}".format(hvd.size()), flush=True)


  # time for data preparation
  prep_stime = time.time()

  # get arg-parse as FLAGS
  FLAGS = get_args(hvd.rank() == 0)

  ## get filenames of training data as list
  train_images_list = []
  for input_datadir in FLAGS.input_datadir:
      train_images_list.extend(glob.glob(os.path.abspath(input_datadir)+'/*.tfrecord'))
  print(len(train_images_list), flush=True)
 
  # make dirs
  os.makedirs(FLAGS.logdir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir, exist_ok=True)

  # loss log filename
  # outputnames
  ctime = datetime.now()
  bname1 = '_nepoch-'+str(FLAGS.num_epoch)+'_lr-'+str(FLAGS.lr)
  bname2 = '_nbatch-'+str(FLAGS.batch_size)+'_lambda'+str(FLAGS.c_lambda)
  ofilename = 'loss_'+FLAGS.expname+bname1+bname2+'_'+str(ctime.strftime("%s"))+'.txt'


  #-----------------------------------------------------
  # Pipeline
  #-----------------------------------------------------
  #### TRAIN
  with tf.device('/GPU'):
    # CLOUD get dataset and one-shot-iterator
    dataset = input_clouds_fn(train_images_list,
                              height=FLAGS.height,
                              width=FLAGS.width,
                              batch_size=FLAGS.batch_size,
                              copy_size=FLAGS.copy_size,
                              distribute=(hvd.size(), hvd.rank()),
                              )
    iterator = iter(dataset)


  ### Resnet model
  if FLAGS.resnet:
      encoder, decoder, autoencoder = model_resnet_fn(
                                            shape=(FLAGS.height,FLAGS.width,FLAGS.channel),
                                            nblocks=FLAGS.nblocks,
                                            base_dim=FLAGS.base_dim,
                                            nstack_layer=FLAGS.nstack_layer)
  ### TGRS2021 model
  else:
      ## add resize layer 128x128 rsesize layer
      encoder, decoder = model_synmetric_resize_fn(
                            shape=(128,128,FLAGS.channel),
                            nblocks=FLAGS.nblocks,
                            base_dim=FLAGS.base_dim,
                            nstack_layer=FLAGS.nstack_layer,
                            rheight=FLAGS.height, rwidth=FLAGS.width)

  if hvd.rank() == 0:
    print("\n {} \n".format(encoder.summary()), flush=True)
    print("\n {} \n".format(decoder.summary()), flush=True)

    ### Loss
    print("\n### ROTATE-INVARIANT LOSS | TRANSFORM FUNCTION ###\n")

  # Number of iteration (for learning rate scheduler)
  num_batches=FLAGS.npatches //FLAGS.batch_size // hvd.size() # npatches is the number of all patches from all files

  # Learning rate decay
  initial_learning_rate = FLAGS.lr   
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=40*num_batches,
    decay_rate=0.96,
    staircase=True,
    name='lr_decay'
  )

  # Apply optimization 
  train_opt = tf.keras.optimizers.SGD(lr_schedule)

  # set-up save models
  save_models = {"encoder": encoder, "decoder": decoder}

  # save model definition
  for m in save_models:
    with open(os.path.join(FLAGS.output_modeldir, m+'.json'), 'w') as f:
      f.write(save_models[m].to_json())


  # End for prep-processing during main code
  prep_etime = (time.time() -prep_stime)/60.0 # minutes
  if hvd.rank() == 0:
    print("\n### Entering Training Loop ###\n")
    print("   Data Pre-Processing time [minutes]  : %f" % prep_etime, flush=True)

  #--------------------------------------------------------------------
  # Restart
  #--------------------------------------------------------------------
  if FLAGS.retrain:
    # e.g. restart_modeldir = os.path.abspath('./output_model/66153901')
    restart_modeldir = os.path.abspath(FLAGS.retrain_datadir)
    for m in save_models:
      gs = load_latest_model_weights(save_models[m],restart_modeldir,m)
  
  # TRAINING

  # train function
  @tf.function
  def train_step(imgs, first_batch=False):
    with tf.GradientTape() as tape:
        
        # RI autoencoder
        trans, rest, sprs=transform_invariant_fn(
                           imgs,
                           encoder,
                           decoder,
                           batch_size=FLAGS.batch_size,
                           degree=FLAGS.degree)


        loss = FLAGS.f_lambda * trans + FLAGS.c_lambda * rest
    
    # --  autoencoder 
    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss, encoder.trainable_weights+decoder.trainable_weights)
    train_opt.apply_gradients(zip(grads, encoder.trainable_weights+decoder.trainable_weights))
    # boradcaset should be done after the first gradient step
    if first_batch:
       hvd.broadcast_variables(encoder.variables, root_rank=0)
       hvd.broadcast_variables(decoder.variables, root_rank=0)
       hvd.broadcast_variables(train_opt.variables(), root_rank=0)
    return loss, trans,rest, sprs


  # custom training loop
  #loss_list = []
  trans_list = []
  rest_list  = []
  sprs_list  = []
  stime = time.time()
  for epoch in range(1,FLAGS.num_epoch+1,1):
      if hvd.rank() == 0: 
        print("\nStart of epoch %d" % (epoch,), flush=True)
      start_time = time.time()
      for step in  range(num_batches):
          try:
              imgs = iterator.get_next() # next(dataset)

              if step ==0 and epoch == 0:
                loss,trans,rest,sprs  = train_step(imgs, first_batch=True)
              else:
                loss,trans,rest,sprs  = train_step(imgs, first_batch=False)

              # summary
              if  hvd.rank() == 0 and step % 100 == 0:

                  print(
                         "iteration {:7} | Transform {:10} | Restration {:10} | Sparse {:10}".format(
                          step, trans, rest, sprs), 
                        flush=True
                  )  
                  trans_list.append(trans)
                  rest_list.append(rest)
                  sprs_list.append(sprs)

          except tf.errors.OutOfRangeError as e:
            if hvd.rank() == 0:
              print(f"End EPOCH {epoch} \n", flush=True) 
            pass        

      # show compute time every epochs
      print(f"Rank  %d   Time taken: %.3fs" % ( hvd.rank(),time.time() - start_time), flush=True)
          
      # save model at every N steps
      if hvd.rank() == 0 and  epoch % FLAGS.save_every == 0 and epoch > 0:
         for m in save_models:
           save_models[m].save_weights(
             os.path.join(
               FLAGS.output_modeldir, "{}-{}.h5".format(m, epoch)
             )
           )

  if hvd.rank() == 0:
    with tf.device("/CPU"):
      # system
      with open(os.path.join(FLAGS.logdir, ofilename), 'w') as f:
        # RI
        for re, ro, rs in zip(trans_list, rest_list, sprs_list):
          f.write(str(re)+','+str(ro)+','+str(rs)+'\n')
      
    print("### TRAINING NORMAL END ###")
    # FINISH
    etime = (time.time() -stime)/60.0 # minutes
    print("   Execution time [minutes]  : %f" % etime, flush=True)
