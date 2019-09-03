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
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    # layer 2
    x = Conv2D(filters=10, kernel_size=3, padding='same')(x)
    x = ReLU()(x)
    #x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
        
             
    # build model for encoder
    encoder = Model(inp, x, name='encoder')
             
    x = inp = Input(x.shape[1:], name="decoder_input")
    
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

def rotate_fn(images, seed=0, return_np=False):
    """
    Apply random rotation to data and parse to dataset module
    images: before parse to encoder

    * copy function from classifier.py
    """
    # in the case numpy is input
    # float point 32 if numpy array
    #if isinstance(images, np.ndarray):
    #  images = images.astype(np.float32)

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
    #imgs  = make_copy_rotate(imgs_tf,batch_size=batch_size,copy_size=copy_size) 
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
    #imgs  = make_copy_rotate(imgs_tf,batch_size=batch_size,copy_size=copy_size) 
    #oimgs = make_copy_rotate(imgs_tf,batch_size=batch_size,copy_size=copy_size, rotate=False) 
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
    # ++ common version
    dataset = tf.data.Dataset.from_tensor_slices((data1))
    dataset = dataset.shuffle(1000).repeat().batch(int(batch_size/copy_size))
    return dataset

def make_copy_rotate_image(oimgs_tf, batch_size=32, copy_size=4):
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
    img_list.extend([tf.reshape(tmp_img_tf, (1,28,28,1))] )
    #print(tmp_img_tf.shape)
    #tmp_img_list.append(tf.expand_dims(tmp_img_tf))
    img_list.extend([ tf.expand_dims(tf.identity(tmp_img_tf), axis=0) for i in range(copy_size-1)])

  coimgs = tf.concat(img_list, axis=0)
  crimgs = rotate_fn(coimgs, seed=np.random.randint(0,999), return_np=False)
  etime = datetime.now()
  print(" make_copy_rotate {} s".format(etime - stime))
  return crimgs, coimgs

#TODO remove this fn as we don't use 
# preprocess fn for batch from dataset
def generic2_make_copy_rotate(oimgs_tf, batch_size=32, copy_size=4, rotate=True):
  """
    INPUT:
      oimgs: original images in minibatch
    OUTPUT:
      crimgs: minibatch with original and these copy + rotations
  """
  # operate within cpu   
  #with tf.device('/cpu:0'):
  stime = datetime.now()
  img_list = []
  for idx in range(int(batch_size/copy_size)):
    tmp_img_list = []
    tmp_img_tf = oimgs_tf[idx]
    tmp_img_list = [ tf.expand_dims(tf.identity(tmp_img_tf), axis=0) for i in range(copy_size)]
    _cimgs = tf.concat(tmp_img_list, axis=0)
    if rotate:
      _crimgs = rotate_fn(_cimgs, seed=np.random.randint(0,999), return_np=False)
      img_list.append(_crimgs)
    else:
      img_list.append(_cimgs)

  #crimgs = np.concatenate(img_list, axis=0)
  crimgs = tf.concat(img_list, axis=0)
  etime = datetime.now()
  print(" make_copy_rotate {} s".format(etime - stime))
  #del tmp_img_list, img_list
  return crimgs

#TODO remove this fn as we don't use 
def generic_make_copy_rotate(oimgs_np, copy_size=4, rotate=True):
  """
   ** OLD VERSION:
      Function was operated by numpy. MUST be changed to tf.tensor for acceleration
    INPUT:
      oimgs: original images in minibatch
    OUTPUT:
      crimgs: minibatch with original and these copy + rotations
  """
  img_list = []
  for img in oimgs_np:
    tmp_img_list = []
    tmp_img_list = [ img.copy().reshape(1,28,28,1) for i in range(copy_size)]
    _cimgs = np.concatenate(tmp_img_list, axis=0)
    if rotate:
      _crimgs = rotate_fn(_cimgs, seed=np.random.randint(0,999), return_np=False)
      img_list.append(_crimgs)
    else:
      img_list.append(_cimgs)

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

  # ad-hoc params
  num_test_images = 10
  
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

  # get model
  encoder, decoder = model_fn()

  # get data from iterator
  dataset = input_fn(mnist.train.images, 
                   batch_size=FLAGS.batch_size, 
                   rotation=FLAGS.rotation,
                   copy_size=FLAGS.copy_size
  )

  # apply preprocessing  
  dataset = dataset.map(lambda x: make_copy_rotate_image(
    x,batch_size=FLAGS.batch_size,copy_size=FLAGS.copy_size)
  )

  # iterator + get next original img
  img , oimg = dataset.make_one_shot_iterator().get_next()

  with tf.device('/CPU'):
    # compute loss and train_ops
    loss_rotate = loss_rotate_fn(img, encoder,
                               batch_size=FLAGS.batch_size,
                               copy_size=FLAGS.copy_size,
                               c_lambda=FLAGS.c_lambda
    )
  
    loss_reconst, reconst_list = loss_reconst_fn(
                               img, oimg,
                               encoder, decoder, 
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
  loss_all = tf.math.add(loss_reconst, loss_rotate)
  train_ops = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss_all)

  # set-up save models
  save_models = {"encoder": encoder, "decoder": decoder}

  # save model definition
  for m in save_models:
    with open(os.path.join(FLAGS.output_modeldir, m+'.json'), 'w') as f:
      f.write(save_models[m].to_json())

  # End for prep-processing during main code
  prep_etime = (time.time() -prep_stime)/60.0 # minutes
  print("\n### Entering Training Loop ###\n")
  print("   Data Pre-Processing time [minutes]  : %f" % prep_etime, flush=True)

  # gpu config 
  config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True
    ),
    log_device_placement=False
  )

  with tf.Session(config=config) as sess:
    # initial run
    init=tf.global_variables_initializer()
    X=tf.placeholder(tf.float32,shape=[None,28,28,1]) 
    sess.run(init)

    # initialize other variables
    train_loss_list = []
    num_batches=int(mnist.train.num_examples/FLAGS.copy_size)//FLAGS.batch_size
    angle_list = [i for i in range(1,360, FLAGS.dangle)]

    # Trace and Profiling options
    run_metadata = tf.RunMetadata()
    run_opts = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)
    #profiler = Profiler(sess.graph)

    #====================================================================
    # Training
    #====================================================================
    stime = time.time()
    for epoch in range(FLAGS.num_epoch):
    #for epoch in range(0,1,1):
        #for iteration in range(5):
        for iteration in range(num_batches):
            ## run all in once
            #_, train_loss_reconst, train_loss_rotate, min_idx, tf_summary = sess.run(
            #  [train_ops, loss_reconst, loss_rotate,tf.math.argmin(reconst_list), merged]
            #)

            ## run separately
            # main total loss
            sess.run(train_ops, run_metadata=run_metadata, options=run_opts)
            # sub individual loss and angle
            with tf.device('/CPU'):
                train_loss_reconst, train_loss_rotate, min_idx = sess.run( 
                    [loss_reconst, loss_rotate,tf.math.argmin(reconst_list)]
                )
                # check angles
                print( "min idx = {} | min angle = {} ".format(min_idx, angle_list[min_idx]) )

                # set for check loss/iteration
                print("iteration {}  loss reconst {}  loss rotate {}".format(
                    iteration, train_loss_reconst, train_loss_rotate), flush=True
                )   

            ## TODO make argparser for save every
            # save scaler summary at every 10 steps
            if iteration % 10 == 0 :
              # summary
              tf_summary = sess.run(merged, run_metadata=run_metadata, options=run_opts )
              summary_writer.add_summary(tf_summary,epoch*num_batches+iteration )
              summary_writer.flush() # write immediately
              # profiling
              #profiler.add_step(epoch*num_batches+iteration, run_metadata)
              #profiler.profile_graph(options=ProfileOptionBuilder(
              #  ProfileOptionBuilder.time_and_memory())
              #  .with_step(epoch*num_batches+iteration)
              #  .build()
              #)
              #profiler.advise({"AcceleratorUtilizationChecker": {}})

            # save model at every N steps
            if iteration % 20 == 0 and iteration != 0:
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
