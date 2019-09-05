#
# + cleaning version: TO BE trian.py
#
import matplotlib
matplotlib.use('Agg')

import os
import gc
import json
import time
import math
import argparse
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.profiler import ProfileOptionBuilder, Profiler
from tensorflow.python import debug as tf_debug

def get_args():
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
    '--output_modeldir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--lr',
    type=float,
    default=0.001
  )
  # comment off
  #p.add_argument(
  #  '--lr_reconst',
  #  type=float,
  #  default=0.001
  #)
  #p.add_argument(
  #  '--lr_rotate',
  #  type=float,
  #  default=0.001
  #)
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
    '--nblocks',
    type=int,
    default=5
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
    '--dangle',
    type=int,
    default=2
  )
  p.add_argument(
    '--c_lambda',
    type=float,
    default=1.0
  )
  p.add_argument(
    '--rotation',
    action="store_true", 
    help='if user attachs this option, training images will be rondamly rotated',
    default=False
  )
  p.add_argument(
    '--debug',
    action='store_true',
  )
  args = p.parse_args()
  for f in args.__dict__:
    print("\t", f, (25 - len(f)) * " ", args.__dict__[f])
  print("\n")
  return args
  

def model_fn(shape=(32,32,1), nblocks=5, base_dim=2) :
    """
      Reference: https://blog.keras.io/building-autoencoders-in-keras.html
    """
    def convSeries_fn(x,
                      filters=16, 
                      kernel_size=3, 
                      nstack_layer=3, 
                      stride=2, 
                      up=True, 
                      pooling=True
                      ):
      """
      INPUT
        nstack_layer : number of iteration of conv layer before pooling
        up           : boolean. True is encoder, False is decoder(conv2D transpose)
      """
      for idx  in range(nstack_layer):
        if up:
          x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                     kernel_initializer='he_normal')(x)
        else:
          if idx == nstack_layer-1:
            x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, 
                                strides=(stride,stride), padding='same')(x)
          else:
            x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                     kernel_initializer='he_normal')(x)
        x = ReLU()(x)
          
      if pooling:
        x = MaxPooling2D((2, 2), padding='same')(x)
      x = BatchNormalization()(x)
      return x

    # set params
    params = {
      'filters': [ 2**(i+base_dim) for i in range(nblocks)],
      'kernel_size': 3
    }
    
    ## start construction

    x = inp = Input(shape=shape, name='encoding_input')
    x = Conv2D(filters=1, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    # encoder layers
    for iblock in range(nblocks):
      filters = params["filters"][iblock]
      kernel_size = params["kernel_size"]
      if iblock != nblocks-1:
        x = convSeries_fn(x,filters=filters, kernel_size=kernel_size, up=True)
      else:
        x = convSeries_fn(x,filters=filters, kernel_size=kernel_size, up=True, pooling=False)
             
    # build model for encoder + digit layer
    encoder = Model(inp, x, name='encoder')
             
    x = inp = Input(x.shape[1:], name="decoder_input")
    # decoder layers
    for iblock in range(nblocks):
      filters = params["filters"][::-1][iblock]
      kernel_size = params["kernel_size"]
      if not iblock == nblocks-1:
        x = convSeries_fn(x,filters=filters, kernel_size=kernel_size, up=False, pooling=False)
      else:
        x = convSeries_fn(x,filters=filters, kernel_size=kernel_size, up=True, pooling=False)
    
    x = Conv2D(filters=1, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    decoder = Model(inp, x, name='decoder')
             
    return encoder, decoder

def resize_image_fn(imgs,height=32, width=32):
  """Resize images from (#batch,H',W',C) --> (#batch,H,W,C)
  """
  reimgs = tf.image.resize_images(imgs, (height, width))
  return reimgs

def rotate_fn(images, seed=0, return_np=False):
    """
    Apply random rotation to data and parse to dataset module
    images: before parse to encoder

    * copy function from classifier.py
    """

    # random rotation
    random_angles = tf.random.uniform(
        shape = (tf.shape(images)[0], ), 
        minval = 0*math.pi/180,
        maxval = 359.999*math.pi/180,
        dtype=tf.float32,
        seed = seed
    )
    rotated_tensor_images = tf.contrib.image.transform(
      images,
      tf.contrib.image.angles_to_projective_transforms(
        random_angles, tf.cast(tf.shape(images)[1], tf.float32), 
            tf.cast(tf.shape(images)[2], tf.float32)
        )
    )
    
    #FIXME debug here
    if return_np:
      rotated_images = tf.keras.backend.eval(rotated_tensor_images)
      return rotated_images
    else:
      return rotated_tensor_images


def loss_rotate_fn(imgs, 
                   encoder,
                   batch_size=32,
                   copy_size=4,
                   c_lambda=1
                   ):

    stime = datetime.now()
    loss_rotate_list = []

    encoded_imgs = encoder(imgs)
    for idx in range(int(batch_size/copy_size)):
      _imgs = encoded_imgs[copy_size*idx:copy_size*(idx+1)]
      _loss_rotate_list = []
      for (i,j) in itertools.combinations([i for i in range(copy_size)],2):
        _loss_rotate_list.append(
          tf.reduce_mean( tf.square( _imgs[i] - _imgs[j]) )
        )
      loss_rotate_list.append(tf.reduce_max(_loss_rotate_list))

    loss_rotate = tf.reduce_mean(tf.stack(loss_rotate_list))

    etime = datetime.now()
    print(" Loss Rotate {} s".format(etime - stime))
    #del loss_rotate_list, _loss_rotate_list, _imgs
    return tf.multiply(tf.constant(c_lambda ,dtype=tf.float32), loss_rotate)

def loss_reconst_fn(imgs,
                    oimgs, 
                    encoder,
                    decoder,
                    batch_size=32,
                    copy_size=4,
                    dangle=2
                    ):

    stime = datetime.now()
    def rotate_operation(imgs, angle=1):
        """angle: Radian.
            angle = degree * math.pi/180
        """
        rimgs = tf.contrib.image.rotate(
                imgs,
                tf.constant(angle ,dtype=tf.float32),
                interpolation='NEAREST',
                name=None
        )
        return rimgs

    loss_reconst_list = []
    angle_list = [i*math.pi/180 for i in range(1,360,dangle)]
     
    # 08/28 2PM  before modification 
    encoded_imgs = encoder(oimgs)
    reconst_imgs = decoder(encoded_imgs)
    for angle in angle_list:
      rimgs = rotate_operation(reconst_imgs,angle=angle) # R_theta(x_hat)
      loss_reconst_list.append(tf.reduce_mean(tf.square(imgs - rimgs)))
    loss_reconst = tf.reduce_min(loss_reconst_list)
    etime = datetime.now()
    print(" Loss Reconst {} s".format(etime - stime))
    return loss_reconst, loss_reconst_list


def input_fn(data, batch_size=32, rotation=False, copy_size=4, height=32, width=32):
    # check batch/copy ratio
    try:
      if batch_size % copy_size == 0:
        print("\n Number of actual original images == {} ".format(int(batch_size/copy_size)))
    except:
      raise ValueError("\n Division of batch size and copy size is not Integer \n")

    data1 = data.reshape(-1,28,28,1)
    if rotation:
      data1 = rotate_fn(data1)
      print(" Apply rondam rotation to training images for AE ")
    # ++ common version
    data1 = resize_image_fn(data1, height=height, width=width)
    dataset = tf.data.Dataset.from_tensor_slices((data1))
    dataset = dataset.shuffle(1000).repeat().batch(int(batch_size/copy_size))
    return dataset

def make_copy_rotate_image(oimgs_tf, batch_size=32, copy_size=4, height=32, width=32):
  """
    INPUT:
      oimgs_tf : original images in minibatch for rotation
    OUTPUT:
      crimgs: minibatch with original and these copy + rotations
  """
  print(" SHAPE in make_copy_rotate_image", oimgs_tf.shape)
  # operate within cpu   
  stime = datetime.now()
  img_list = []
  for idx in range(int(batch_size/copy_size)):
    tmp_img_tf = oimgs_tf[idx]
    img_list.extend([tf.reshape(tmp_img_tf, (1,height,width,1))] )
    img_list.extend([ tf.expand_dims(tf.identity(tmp_img_tf), axis=0) for i in range(copy_size-1)])

  coimgs = tf.concat(img_list, axis=0)
  crimgs = rotate_fn(coimgs, seed=np.random.randint(0,999), return_np=False)
  etime = datetime.now()
  print(" make_copy_rotate {} s".format(etime - stime))
  return crimgs, coimgs

if __name__ == '__main__':
  # time for data preparation
  prep_stime = time.time()

  # set data
  mnist = input_data.read_data_sets(os.path.abspath("./MNIST_data/"), one_hot=True)

  # get arg-parse as FLAGS
  FLAGS = get_args()

  # ad-hoc params
  num_test_images = int(FLAGS.batch_size/FLAGS.copy_size)
  
  # make dirs
  os.makedirs(FLAGS.logdir, exist_ok=True)
  os.makedirs(FLAGS.figdir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir+'/timelines', exist_ok=True)

  # outputnames
  bname1 = '_nepoch-'+str(FLAGS.num_epoch)+'_lr-'+str(FLAGS.lr)
  bname2 = '_nbatch-'+str(FLAGS.batch_size)+'_lambda'+str(FLAGS.c_lambda)+'_dangle'+str(FLAGS.dangle)
  figname   = 'fig_'+FLAGS.expname+bname1+bname2
  ofilename = 'loss_'+FLAGS.expname+bname1+bname2+'.txt'

  # set global time step
  global_step = tf.train.get_or_create_global_step()

  with tf.device('/CPU'):
    # get dataset and one-shot-iterator
    dataset = input_fn(mnist.train.images, 
                     batch_size=FLAGS.batch_size, 
                     rotation=FLAGS.rotation,
                     copy_size=FLAGS.copy_size
    )
    # apply preprocessing  
    dataset_mapper = dataset.map(lambda x: make_copy_rotate_image(
            x,batch_size=FLAGS.batch_size,copy_size=FLAGS.copy_size,
            height=FLAGS.height,width=FLAGS.width
        )
    )

  # why get_one_shot_iterator leads OOM error?
  train_iterator = dataset_mapper.make_initializable_iterator()
  imgs, oimgs  = train_iterator.get_next()

  # get model
  encoder, decoder = model_fn(nblocks=FLAGS.nblocks)
  #print("\n {} \n".format(encoder.summary()), flush=True)
  #print("\n {} \n".format(decoder.summary()), flush=True)

  # loss + optimizer
  # compute loss and train_ops
  loss_rotate = loss_rotate_fn(imgs, encoder,
                             batch_size=FLAGS.batch_size,
                             copy_size=FLAGS.copy_size,
                             c_lambda=FLAGS.c_lambda
  )
  
  loss_reconst, reconst_list = loss_reconst_fn(
                             imgs, oimgs,
                             encoder, decoder, 
                             batch_size=FLAGS.batch_size,
                             copy_size=FLAGS.copy_size,
                             dangle=FLAGS.dangle
  )
 
  # observe loss values with tensorboard
  with tf.name_scope("summary"):
    tf.summary.scalar("reconst loss", loss_reconst)
    tf.summary.scalar("rotate loss", loss_rotate)
    merged = tf.summary.merge_all()

  # Apply optimization
  # ++ Method 1
  # Method 2 sometimes showed 0 loss in early statge. 
  loss_all = tf.math.add(loss_reconst, loss_rotate)
  train_ops = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss_all)

  # ++ Method 2
  #train_ops_reconst = tf.train.AdamOptimizer(FLAGS.lr_reconst).minimize(loss_reconst)
  #train_ops_rotate  = tf.train.AdamOptimizer(FLAGS.lr_rotate).minimize(loss_rotate)
  #train_ops = tf.group(train_ops_reconst,train_ops_rotate)

  # set-up save models
  save_models = {"encoder": encoder, "decoder": decoder}

  # save model definition
  for m in save_models:
    with open(os.path.join(FLAGS.output_modeldir, m+'.json'), 'w') as f:
      f.write(save_models[m].to_json())

  # gpu config 
  config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True
    ),
    log_device_placement=False
  )

  # End for prep-processing during main code
  prep_etime = (time.time() -prep_stime)/60.0 # minutes
  print("\n### Entering Training Loop ###\n")
  print("   Data Pre-Processing time [minutes]  : %f" % prep_etime, flush=True)
  
  # TRAINING
  with tf.Session(config=config) as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # initial run
    init=tf.global_variables_initializer()
    sess.run(init)
    sess.run(train_iterator.initializer)

    # initialize other variables
    train_loss_reconst_list = []
    train_loss_rotate_list = []
    num_batches=int(mnist.train.num_examples/FLAGS.copy_size)//FLAGS.batch_size
    angle_list = [i for i in range(1,360, FLAGS.dangle)]

    # Trace and Profiling options
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.output_modeldir, 'logs'), sess.graph) 
    run_metadata = tf.RunMetadata()
    run_opts = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)

    #====================================================================
    # Training
    #====================================================================
    stime = time.time()
    for epoch in range(FLAGS.num_epoch):
        for iteration in range(num_batches):
            ## run all in once
            _, train_loss_reconst, train_loss_rotate, min_idx, tf_summary = sess.run(
              [train_ops, loss_reconst, loss_rotate,tf.math.argmin(reconst_list), merged]
              , run_metadata=run_metadata, options=run_opts
            )
            # check angles
            print( "min idx = {} | min angle = {} ".format(min_idx, angle_list[min_idx]) )

            # set for check loss/iteration
            print("iteration {}  loss reconst {}  loss rotate {}".format(
                iteration, train_loss_reconst, train_loss_rotate), flush=True
            )
            train_loss_reconst_list.append(train_loss_reconst)
            train_loss_rotate_list.append(train_loss_rotate)


            ## TODO make argparser for save every
            if iteration % 20 == 0 :
              # summary
              total_iteration = epoch*num_batches + iteration
              summary_writer.add_run_metadata(run_metadata, 'step%05d' % total_iteration)
              summary_writer.add_summary(tf_summary, total_iteration )
              summary_writer.flush() # write immediately
              # profiling
              #profiler.add_step(epoch*num_batches+iteration, run_metadata)
              #profiler.profile_graph(options=ProfileOptionBuilder(
              #  ProfileOptionBuilder.time_and_memory())
              #  .with_step(epoch*num_batches+iteration)
              #  .build()
              #)
              #profiler.advise({"AcceleratorUtilizationChecker": {}})

              #============================================================
              #   Chrome Profiler
              #============================================================
              fetched_timeline = timeline.Timeline(run_metadata.step_stats)
              chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=True)
              with open(FLAGS.output_modeldir+'/timelines/time%d-%d.json' % (epoch, iteration), 'w') as f:
                f.write(chrome_trace)

            # save model at every N steps
            if iteration % 20 == 0 and iteration != 0:
              if epoch % FLAGS.save_every == 0:
                for m in save_models:
                  save_models[m].save_weights(
                    os.path.join(
                      FLAGS.output_modeldir, "{}-{}-iter{}.h5".format(m, epoch, iteration)
                    )
                  ) 
        
        # save model at every N steps
        if epoch % FLAGS.save_every == 0:
          for m in save_models:
            save_models[m].save_weights(
              os.path.join(
                FLAGS.output_modeldir, "{}-{}.h5".format(m, epoch)
              )
            )

    # save loss result
    with tf.device("/CPU"):
      encoded = encoder(imgs)
      results = decoder(encoded).eval()
      
      test_images = oimgs.eval()
      #Comparing original images with reconstructions
      f,a=plt.subplots(2,num_test_images,figsize=(2*num_test_images,4))
      for idx, i in enumerate(range(0, FLAGS.batch_size, FLAGS.copy_size)):
        a[0][idx].imshow(np.reshape(test_images[i],(FLAGS.height,FLAGS.width)), cmap='jet')
        a[1][idx].imshow(np.reshape(results[i],(FLAGS.height,FLAGS.width)), cmap='jet')
        # set axis turn off
        a[0][idx].set_xticklabels([])
        a[0][idx].set_yticklabels([])
        a[1][idx].set_xticklabels([])
        a[1][idx].set_yticklabels([])
      plt.savefig(FLAGS.figdir+'/'+figname+'.png')
    
      with open(os.path.join(FLAGS.logdir, ofilename), 'w') as f:
        f.write(' Reconst loss  &  Rotate loss\n')
        for re,ro in zip(train_loss_reconst_list,train_loss_rotate_list):
          f.write(str(re)+','+str(ro)+'\n')
        
  # FINISH
  etime = (time.time() -stime)/60.0 # minutes
  print("### NORMAL END ###")
  print("   Execution time [minutes]  : %f" % etime, flush=True)
