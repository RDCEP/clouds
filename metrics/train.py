
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
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.profiler import ProfileOptionBuilder, Profiler

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
    '--expname',
    type=str,
    default='new'
  )
  p.add_argument(
    '--lr',
    type=float,
    default=0.001
  )
  p.add_argument(
    '--lr_reconst',
    type=float,
    default=0.001
  )
  p.add_argument(
    '--lr_rotate',
    type=float,
    default=0.001
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
    '--save_every',
    type=int,
    default=10
  )
  p.add_argument(
    '--rotation',
    action="store_true", 
    help='if user attachs this option, training images will be rondamly rotated',
    default=False
  )
  args = p.parse_args()
  for f in args.__dict__:
    print("\t", f, (25 - len(f)) * " ", args.__dict__[f])
  print("\n")
  return args
  

def model_fn(shape=(28,28,1)) :
    """
      Reference: https://blog.keras.io/building-autoencoders-in-keras.html
    """
    x = inp = Input(shape=shape, name='encoding_input')
    
    # layer 1
    x = Conv2D(filters=16, kernel_size=3, padding='same',
              kernel_initializer='he_normal')(x)
    x = ReLU()(x)
    #x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    # layer 2
    x = Conv2D(filters=10, kernel_size=3, padding='same')(x)
    x = ReLU()(x)
    #x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
        
    # layer 3
    #x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
    #x = LeakyReLU()(x)
    #x = MaxPooling2D((2, 2), padding='same')(x)
    #x = BatchNormalization()(x)
             
    # build model for encoder
    encoder = Model(inp, x, name='encoder')
             
    x = inp = Input(x.shape[1:], name="decoder_input")
    # layer 4
    #x = Conv2DTranspose(filters=32, kernel_size=3,strides=(2,2),padding='same')(x)
    #x = LeakyReLU()(x)
    #x = BatchNormalization()(x)
    
    # layer 5
    x = Conv2DTranspose(filters=10, kernel_size=3,strides=(2,2), padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
             
    # layer 4
    x = Conv2DTranspose(filters=16, kernel_size=3,strides=(2,2), padding='same')(x)
    x = LeakyReLU()(x)
    
    # layer 5
    x = Conv2D(filters=1, kernel_size=3, padding='same')(x)
    decoder = Model(inp, x, name='decoder')
             
    return encoder, decoder

def rotate_fn(images, seed=0, return_np=True):
    """
    Apply random rotation to data and parse to dataset module
    images: before parse to encoder

    * copy function from classifier.py
    """
    # float point 32 if numpy array
    if isinstance(images, np.ndarray):
      images = images.astype(np.float32)

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
    
    if return_np:
      # convert from tensor to numpy
      sess = tf.Session()
      with sess.as_default():
        rotated_images = rotated_tensor_images.eval()
      return rotated_images
    else:
      return rotated_tensor_images


def loss_rotate_fn(imgs, 
                   encoder,
                   batch_size=32,
                   copy_size=4,
                   c_lambda=1
                   ):
    shape = (-1,28,28,1)
    loss_rotate_list = []
    for idx in range(int(batch_size/copy_size)):
      _imgs = imgs[copy_size*idx:copy_size*(idx+1)]
      _loss_rotate_list = []
      for (i,j) in itertools.combinations([i for i in range(copy_size)],2):
        _loss_rotate_list.append(
          tf.reduce_mean(
              tf.square( encoder(tf.reshape(_imgs[i],shape)) - encoder(tf.reshape(_imgs[j],shape)) )
          ) 
        )
      loss_rotate_list.append(tf.reduce_max(_loss_rotate_list))

    loss_rotate = tf.reduce_mean(tf.stack(loss_rotate_list))

    return tf.multiply(tf.constant(c_lambda ,dtype=tf.float32), loss_rotate)

def loss_reconst_fn(imgs, 
                    encoder,
                    decoder,
                    batch_size=32,
                    copy_size=4,
                    dangle=2
                    ):

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
    encoded_imgs = encoder(imgs)
    for angle in angle_list:
      rimgs = rotate_operation(decoder(encoded_imgs),angle=angle) # R_theta(x_hat)
      loss_reconst_list.append(tf.reduce_mean(tf.square(imgs - rimgs)))
    loss_reconst = tf.reduce_min(tf.stack(loss_reconst_list))
    min_idx = tf.keras.backend.eval(tf.math.argmin(loss_reconst_list)) 
    print( "min idx = {} | min angle = {} ".format(min_idx, angle_list[min_idx]) )

    return loss_reconst


def input_fn(data, batch_size=32, rotation=False, copy_size=4):
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
    dataset = tf.data.Dataset.from_tensor_slices((data1))
    dataset = dataset.shuffle(1000).repeat().batch(int(batch_size/copy_size))
    return dataset

def make_copy_rotate(oimgs, copy_size=4):
  """
    INPUT:
      oimgs: original images in minibatch
    OUTPUT:
      crimgs: minibatch with original and these copy + rotations
  """
  # tensor to numpy
  sess = tf.Session()
  with sess.as_default():
    oimgs_np = oimgs.eval()

  img_list = []
  for img in oimgs_np:
    tmp_img_list = []
    tmp_img_list = [ img.copy().reshape(1,28,28,1) for i in range(copy_size)]
    _cimgs = np.concatenate(tmp_img_list, axis=0)
    _crimgs = rotate_fn(_cimgs, seed=np.random.randint(0,999), return_np=False)
    img_list.append(_crimgs)

  #crimgs = np.concatenate(img_list, axis=0)
  crimgs = tf.concat(img_list, axis=0)
  return crimgs
    

if __name__ == "__main__":
  # time for data preparation
  prep_stime = time.time()

  # set data
  mnist = input_data.read_data_sets(os.path.abspath("./MNIST_data/"), one_hot=True)

  # get arg-parse as FLAGS
  FLAGS = get_args()
  # make dirs
  os.makedirs(FLAGS.logdir, exist_ok=True)
  os.makedirs(FLAGS.figdir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir+'/timelines', exist_ok=True)

  # ad-hoc params
  num_test_images = 10

  # get model
  encoder, decoder = model_fn()

  dataset = input_fn(mnist.train.images, 
                     batch_size=FLAGS.batch_size, 
                     rotation=FLAGS.rotation,
                     copy_size=FLAGS.copy_size
  )
  img_beforeCopyRotate = dataset.make_one_shot_iterator().get_next()

  # convert imgs to imgs with copy of rotations
  img = make_copy_rotate(img_beforeCopyRotate,copy_size=FLAGS.copy_size)

  # compute loss and train_ops
  loss_rotate = loss_rotate_fn(img, encoder,
                               batch_size=FLAGS.batch_size,
                               copy_size=FLAGS.copy_size,
                               c_lambda=FLAGS.c_lambda
  )
  loss_reconst = loss_reconst_fn(img, encoder, decoder, 
                                 batch_size=FLAGS.batch_size,
                                 copy_size=FLAGS.copy_size,
                                 dangle=FLAGS.dangle
  )
  gc.collect()
 
  # observe loss values with tensorboard
  with tf.name_scope("summary"):
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.output_modeldir, 'logs')) 
    tf.summary.scalar("reconst loss", loss_reconst)
    tf.summary.scalar("rotate loss", loss_rotate)
    merged = tf.summary.merge_all()

  # Apply optimization
  # Method 2: Apply Adam concurrently
  #  This method's accuracy was so bad when lambda is too big s.t. >10
  loss_all = tf.math.add(loss_reconst, loss_rotate)
  train_ops  = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss_all)

  # Method 1: Apply Adam individually
  #train_ops_reconst = tf.train.AdamOptimizer(FLAGS.lr_reconst).minimize(loss_reconst)
  #train_ops_rotate = tf.train.AdamOptimizer(FLAGS.lr_rotate).minimize(loss_rotate)
  #train_ops = tf.group(train_ops_reconst, train_ops_rotate)

  # set-up save models
  save_models = {"encoder": encoder, "decoder": decoder}

  # save model definition
  for m in save_models:
    with open(os.path.join(FLAGS.output_modeldir, m+'.json'), 'w') as f:
      f.write(save_models[m].to_json())

  #====================================================================
  # Training
  #====================================================================

  # initialize
  init=tf.global_variables_initializer()
  X=tf.placeholder(tf.float32,shape=[None,28,28,1])
  train_loss_list = []
  
  # outputnames
  #bname1 = 'nepoch-'+str(FLAGS.num_epoch)+'_lr-'+str(FLAGS.lr)
  bname1 = '_nepoch-'+str(FLAGS.num_epoch)+'_lr-'+str(FLAGS.lr)
  bname2 = '_nbatch-'+str(FLAGS.batch_size)+'_lambda'+str(FLAGS.c_lambda)+'_dangle'+str(FLAGS.dangle)
  figname   = 'fig_'+FLAGS.expname+bname1+bname2
  ofilename = 'loss_'+FLAGS.expname+bname1+bname2+'.txt'

  # Trace and Profiling options
  run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  # End for prep-processing during main code
  prep_etime = (time.time() -prep_stime)/60.0 # minutes
  print("\n### Entering Training Loop ###\n")
  print("   Data Pre-Processing time [minutes]  : %f" % prep_etime, flush=True)

  # start!
  stime = time.time()
  with tf.Session() as sess:
    # initial run
    sess.run(init, options=run_opts, run_metadata=run_metadata)

    # enter training loop
    for epoch in range(FLAGS.num_epoch):
        #num_batches=mnist.train.num_examples//FLAGS.batch_size
        num_batches=int(mnist.train.num_examples/FLAGS.copy_size)//FLAGS.batch_size
        for iteration in range(num_batches):
            #sess.run(train_ops,options=run_opts, run_metadata=run_metadata)
            _, tf_summary = sess.run([train_ops, merged],options=run_opts, run_metadata=run_metadata)

            # set for debug
            X_batch,y_batch=mnist.train.next_batch(FLAGS.batch_size)
            train_loss_reconst= loss_reconst.eval(feed_dict={X:X_batch.reshape(-1,28,28,1)})
            train_loss_rotate = loss_rotate.eval(feed_dict={X:X_batch.reshape(-1,28,28,1)})
            print("iteration {}  loss reconst {}  loss rotate {}".format(
              iteration, train_loss_reconst, train_loss_rotate), flush=True
            )   
            # save scaler summary at every 10 steps
            if iteration % 10 == 0:
              summary_writer.add_summary(tf_summary, _)
              summary_writer.flush() # write immediately
        
            # save model at every N steps
            if iteration % 50 == 0:
              if epoch % FLAGS.save_every == 0:
                for m in save_models:
                  save_models[m].save_weights(
                    os.path.join(
                      FLAGS.output_modeldir, "{}-{}-iter{}.h5".format(m, epoch, iteration)
                    )
                  )
        
        
        X_batch,y_batch=mnist.train.next_batch(FLAGS.batch_size)
        train_loss_reconst= loss_reconst.eval(feed_dict={X:X_batch.reshape(-1,28,28,1)})
        train_loss_rotate = loss_rotate.eval(feed_dict={X:X_batch.reshape(-1,28,28,1)})
        print("epoch {}  loss reconst {}  loss rotate {}".format(
          epoch, train_loss_reconst, train_loss_rotate), flush=True
        )   
        train_loss_list.append(str(train_loss_reconst)+','+str(train_loss_rotate))
    
        # save model at every N steps
        if epoch % FLAGS.save_every == 0:
          for m in save_models:
            save_models[m].save_weights(
              os.path.join(
                FLAGS.output_modeldir, "{}-{}.h5".format(m, epoch)
              )
            )

          #============================================================
          #   Profiler
          #============================================================
          #   Profiler
          fetched_timeline = timeline.Timeline(run_metadata.step_stats)
          chrome_trace = fetched_timeline.generate_chrome_trace_format()
          with open(FLAGS.output_modeldir+'/timelines/time%d.json' % epoch, 'w') as f:
            f.write(chrome_trace)


    # Inference
    encoded=encoder(
        mnist.test.images[:num_test_images].reshape(-1,28,28,1)
    )
    results = decoder(encoded).eval()
    #Comparing original images with reconstructions
    f,a=plt.subplots(2,10,figsize=(20,4))
    for i in range(num_test_images):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)), cmap='jet')
        a[1][i].imshow(np.reshape(results[i],(28,28)), cmap='jet')
        # set axis turn off
        a[0][i].set_xticklabels([])
        a[0][i].set_yticklabels([])
        a[1][i].set_xticklabels([])
        a[1][i].set_yticklabels([])
    plt.savefig(FLAGS.figdir+'/'+figname+'.png')
    
    # save loss result
    with open(os.path.join(FLAGS.logdir, ofilename), 'w') as f:
      for ie in train_loss_list:
        f.write(str(ie)+'\n')

  # FINISH
  etime = (time.time() -stime)/60.0 # minutes
  print("### NORMAL END ###")
  print("   Execution time [minutes]  : %f" % etime, flush=True)
