#
# + cleaning version: TO BE trian.py
#
# + History
# 09/10 Regulate rotation angle only {0, 120, 240}
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
    default=1
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
  

def model_fn(shape=(32,32,1), nblocks=5, base_dim=3) :
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
        maxval = 359.99*math.pi/180,
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
    # remove numpy option and integrate output as tf.tensor
    if return_np:
      rotated_images = tf.keras.backend.eval(rotated_tensor_images)
      return rotated_images
    else:
      return rotated_tensor_images

def rotate_operation(imgs, angle=1):
    """angle: Radian.
        angle = degree * math.pi/180
    """
    rimgs = tf.contrib.image.transform(
    imgs,
    tf.contrib.image.angles_to_projective_transforms(
        angle, tf.cast(tf.shape(imgs)[1], tf.float32), 
        tf.cast(tf.shape(imgs)[2], tf.float32)
        )
    )
    return rimgs

def loss_rotate_fn(imgs, 
                   encoder,
                   batch_size=32,
                   copy_size=4,
                   c_lambda=1,
                   dangle=1
                   ):

    stime = datetime.now()
    loss_rotate_list = []

    # ++ developing version for speed up
    encoded_imgs = encoder(imgs)
    for idx in range(int(batch_size/copy_size)):
      _imgs = encoded_imgs[copy_size*idx:copy_size*(idx+1)]
      #_loss_rotate_list = []
      loss_rotate = 0.0000
      # sum up loss for each image class
      for i,j in itertools.product(range(copy_size), range(copy_size)):
        if i != j:
          loss_rotate += tf.square( _imgs[i] - _imgs[j])
          #_loss_rotate_list.append(tf.square( _imgs[i] - _imgs[j]))
      #loss_rotate_list.append(tf.reduce_sum(tf.stack(_loss_rotate_list, axis=0)))
      loss_rotate_list.append(loss_rotate)

    # loss
    # multiply before reduce_mean
    #loss_rotate_tf = tf.multiply(
    #                      tf.constant(c_lambda ,dtype=tf.float32), 
    #                      tf.stack(loss_rotate_list, axis=0)
    #)
    loss_rotate_tf = tf.stack(loss_rotate_list, axis=0)

    # ++ previous version for speed up
    #for idx in range(int(batch_size/copy_size)):
    #  _imgs = encoded_imgs[copy_size*idx:copy_size*(idx+1)]
    #  _loss_rotate_list = []
    #  for (i,j) in itertools.combinations([i for i in range(copy_size)],2):
    #    _loss_rotate_list.append(
    #      tf.reduce_mean( tf.square( _imgs[i] - _imgs[j]) )
    #    )
    #  loss_rotate_list.append(tf.reduce_max(tf.stack(_loss_rotate_list, axis=0)))
    #loss_rotate = tf.reduce_mean(tf.stack(loss_rotate_list, axis=0))
    #loss_rotate = tf.reduce_max(tf.stack(loss_rotate_list, axis=0))


    etime = datetime.now()
    print(" Loss Rotate {} s".format(etime - stime))
    #return tf.multiply(tf.constant(c_lambda ,dtype=tf.float32), loss_rotate)
    return loss_rotate_tf

def loss_reconst_fn(imgs,
                    encoder,
                    decoder,
                    batch_size=32,
                    copy_size=4,
                    dangle=1
                    ):

    stime = datetime.now()
    loss_reconst_list = []
    angle_list = [i*math.pi/180 for i in range(0,360,dangle)]
     
    encoded_imgs = encoder(imgs)
    decoded_imgs = decoder(encoded_imgs)

    #TODO: Add here to check each image has different theta value
    for angle in angle_list:
      rimgs = rotate_operation(decoded_imgs,angle=angle) # R_theta(x_hat)
      loss_reconst_list.append(
          tf.reduce_mean(tf.square(imgs - rimgs), axis=[1,2,3])
      ) # take mean for each image
      #    tf.reduce_mean(tf.square(imgs - rimgs), axis=[1,2,3])
      # take sum
      #    tf.reduce_sum(tf.square(imgs - rimgs))
    # save optimal theta
    loss_reconst_thetas = tf.math.argmin(tf.stack(loss_reconst_list, axis=0),axis=0)

    # 09/16 version
    #loss_reconst_tf = tf.stack(loss_reconst_list, axis=0)

    # 09/17 devlopment
    loss_reconst_tf = tf.reduce_min(tf.stack(loss_reconst_list, axis=0), axis=0)
    etime = datetime.now()
    print(" Loss Reconst {} s".format(etime - stime))
    #return loss_reconst_tf

    # debug version
    return loss_reconst_tf, loss_reconst_thetas


def input_fn(data, batch_size=32, rotation=False, copy_size=4, height=32, width=32, prefetch=1):
    """
      INPUT:
        prefetch: tf.int64. How many "minibatch" we asynchronously prepare on CPU ahead of GPU
    """
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
    dataset = dataset.shuffle(1000).repeat().batch(int(batch_size/copy_size)).prefetch(prefetch)
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

def loss_L2_fn(imgs, encoder, decoder):
  encoded_imgs = encoder(imgs)
  decoded_imgs = decoder(encoded_imgs)
  return tf.reduce_mean(
            tf.square(imgs - decoded_imgs)
         )

if __name__ == '__main__':
  # time for data preparation
  prep_stime = time.time()

  # get arg-parse as FLAGS
  FLAGS = get_args()

  # set data
  mnist = input_data.read_data_sets(os.path.abspath("./MNIST_data/"), one_hot=False)
  train_images  = mnist.train.images

  # ad-hoc params
  num_test_images = int(FLAGS.batch_size/FLAGS.copy_size)
  #num_test_images = FLAGS.batch_size
  
  # make dirs
  os.makedirs(FLAGS.logdir, exist_ok=True)
  os.makedirs(FLAGS.figdir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir+'/timelines', exist_ok=True)

  # outputnames
  ctime = datetime.now()
  bname1 = '_nepoch-'+str(FLAGS.num_epoch)+'_lr-'+str(FLAGS.lr)
  bname2 = '_nbatch-'+str(FLAGS.batch_size)+'_lambda'+str(FLAGS.c_lambda)+'_dangle'+str(FLAGS.dangle)
  #figname   = 'fig_'+FLAGS.expname+bname1+bname2
  # TODO Add for debug. Remove finally
  figname   = 'fig_'+FLAGS.expname+bname1+bname2+str(ctime.strftime("%s"))
  ofilename = 'loss_'+FLAGS.expname+bname1+bname2+str(ctime.strftime("%s"))+'.txt'
  dfilename = 'degree_'+FLAGS.expname+bname1+bname2+str(ctime.strftime("%s"))+'.txt'

  # set global time step
  global_step = tf.train.get_or_create_global_step()

  with tf.device('/CPU'):
    # get dataset and one-shot-iterator
    dataset = input_fn(train_images, 
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
  loss_rotate  = loss_rotate_fn(imgs, encoder,
                             batch_size=FLAGS.batch_size,
                             copy_size=FLAGS.copy_size,
                             c_lambda=FLAGS.c_lambda
  )
  
  loss_reconst, theta_reconst  = loss_reconst_fn(imgs,
                                  encoder, decoder, 
                                  batch_size=FLAGS.batch_size,
                                  copy_size=FLAGS.copy_size,
                                  dangle=FLAGS.dangle
  )
 
  # debug 1st process
  #loss = loss_L2_fn(imgs,encoder, decoder)

  # Apply optimization 
  # Full version
  train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(
    tf.math.add(
        tf.reduce_mean(loss_reconst),
        tf.multiply(tf.constant(FLAGS.c_lambda, dtype=tf.float32),tf.reduce_mean(loss_rotate))
    )
  )
  #      tf.reduce_mean(loss_rotate)

  # L2 loss: Check Minibatch creation
  #train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(loss)
 

  # Reconst-agn version
  # 09/16
  #train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(
  #      tf.reduce_min(loss_reconst),
  #)
  # 09/17 method 2 take sum as exactly following the equation
  # sum does not work well
  #train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(
  #      tf.reduce_sum(loss_reconst),
  #)
  # 09/17 method-1 take mean
  #train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(
  #      tf.reduce_mean(loss_reconst),
  #)

  # Bottleneck version
  #train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(
  #      tf.multiply( tf.constant(FLAGS.c_lambda, dtype=tf.float32), tf.reduce_mean(loss_rotate))
  #)


  # observe loss values with tensorboard
  with tf.name_scope("summary"):
    #tf.summary.scalar("L2 loss", loss )
    ###tf.summary.scalar("reconst loss", tf.reduce_sum(loss_reconst) )
    tf.summary.scalar("reconst loss", tf.reduce_mean(loss_reconst) )
    #tf.summary.scalar("rotate loss", tf.reduce_mean(loss_rotate) )
    tf.summary.scalar("rotate loss",  
        tf.multiply(tf.constant(FLAGS.c_lambda,dtype=tf.float32), tf.reduce_mean(loss_rotate))
    )
    ###tf.summary.scalar("rotate loss", tf.reduce_sum(loss_rotate) )
    #tf.summary.scalar("rotate loss", tf.reduce_max(loss_rotate) )
    merged = tf.summary.merge_all()

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
    num_batches=int(len(train_images)*FLAGS.copy_size)//FLAGS.batch_size
    angle_list = [i for i in range(0,360, FLAGS.dangle)]
    #angle_list = [i for i in range(0,180, FLAGS.dangle)]
    loss_l2_list = []
    loss_reconst_list = []
    loss_rotate_list = []
    deg_reconst_list = []
    #deg_rotate_list = []

    # Trace and Profiling options
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.output_modeldir, 'logs'), sess.graph) 
    run_metadata = tf.RunMetadata()
    run_opts = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)

    #====================================================================
    # Training
    #====================================================================
    stime = time.time()
    for epoch in range(FLAGS.num_epoch):
    #for epoch in range(0,1,1):
      for iteration in range(num_batches):
      #for iteration in range(0,101,1):
        _, tf.summary = sess.run([train_ops, merged])

        if iteration % 100 == 0:
          _loss_reconst,_loss_rotate, _theta_reconst = sess.run(
              [loss_reconst, loss_rotate, theta_reconst]
          )
          ### rotate 
          #_loss_rotate = sess.run(loss_rotate)
          #print(_loss_rotate)
          ### reconst
          #_loss_reconst, _theta_reconst = sess.run(
          #    [loss_reconst, theta_reconst]
          #)
          #print(_loss_reconst.shape)
          #print(_loss_reconst)
          ### L2
          #_loss_l2 = sess.run(loss)
  
          print(
                 "iteration {:7} | loss reconst {:10}  loss rotate {:10} | Theta 1st term {}".format(
              iteration,   
              math.floor(np.mean(_loss_reconst)*(10**6))/10**6, 
              FLAGS.c_lambda*np.mean(_loss_rotate), 
             _theta_reconst
            ), flush=True
          )
          #    np.mean(_loss_rotate), 

          ### Reconst
          #print(
          #       "iteration {:7} loss reconst {:12} Thetas {}".format(
          #    iteration,np.mean(_loss_reconst), _theta_reconst
          #  ), flush=True
          #)
          #    iteration,np.mean(_loss_reconst), _theta_reconst
          #    iteration,np.min(_loss_reconst), _theta_reconst

          ### Bottleneck
          #print(
          #       "iteration {:7} loss bottleneck {:12} ".format(
          #    iteration, FLAGS.c_lambda*np.mean(_loss_rotate)
          #  ), flush=True
          #)


          #print(
          #       "iteration {:7} | loss {:12} ".format(
          #        iteration, _loss_l2), flush=True
          #)

          # Save loss to lsit
          #loss_l2_list.append(_loss_l2)
          #loss_reconst_list.append(np.sum(_loss_reconst))
          loss_reconst_list.append(np.mean(_loss_reconst))
          loss_rotate_list.append(FLAGS.c_lambda*np.mean(_loss_rotate))
          #loss_rotate_list.append(np.mean(_loss_rotate))
          deg_reconst_list.append(_loss_reconst)

        #if iteration % 2000 == 0:
        #  for m in save_models:
        #    save_models[m].save_weights(
        #      os.path.join(
        #        FLAGS.output_modeldir, "{}-{}-iter{}.h5".format(m, epoch, iteration)
        #      )
        #    ) 
      # save model at every N steps
      if epoch % FLAGS.save_every == 0:
         for m in save_models:
           save_models[m].save_weights(
             os.path.join(
               FLAGS.output_modeldir, "{}-{}.h5".format(m, epoch)
             )
           )

         # correct theta
         #_loss_reconst,_theta_reconst = sess.run(
         #    [loss_reconst, theta_reconst]
         #)
         #print( "\n Save Model Epoch {}: Loss {:12} Theta {} \n".format(
         #     epoch, np.mean(_loss_reconst), _theta_reconst
         #   )
         #)

         #_loss_rotate  = sess.run(loss_rotate)
         #print( "\n Save Model Epoch {}: Loss {:12} \n".format(
         #     epoch, np.mean(_loss_rotate)
         #   )
         #)

         # Full
         _loss_reconst,_loss_rotate, _theta_reconst = sess.run(
              [loss_reconst, loss_rotate, theta_reconst]
         )
         print( "\n Save Model Epoch {}: \n 1st term Loss: {} 2nd term Loss: {} | Correct Thetas: {}  \n".format(
              epoch, np.mean(_loss_reconst), np.mean(_loss_rotate), _theta_reconst
            ),
            flush=True
         )
         #     epoch, np.min(_loss_reconst), _theta_reconst

         #_loss_l2 = sess.run(loss)
         #     #angle_list[np.argmax(_loss_rotate)],
         #print( "\n Save Model Epoch {}: Loss {:12} \n".format(
         #     epoch,
         #     _loss_l2,
         #   )
         #)
         

    #=======================
    # + Visualization
    #=======================
    with tf.device("/CPU"):
      results, test_images, rtest_images = sess.run(
        [decoder(encoder(imgs)), oimgs, imgs]
      )

      #Comparing original images with reconstructions
      f,a=plt.subplots(3,num_test_images,figsize=(2*num_test_images,6))
      for idx, i in enumerate(range(num_test_images)):
        a[0][idx].imshow(np.reshape(test_images[i],(FLAGS.height,FLAGS.width)), cmap='jet')
        a[1][idx].imshow(np.reshape(rtest_images[i],(FLAGS.height,FLAGS.width)), cmap='jet')
        a[2][idx].imshow(np.reshape(results[i],(FLAGS.height,FLAGS.width)), cmap='jet')
        # set axis turn off
        a[0][idx].set_xticklabels([])
        a[0][idx].set_yticklabels([])
        a[1][idx].set_xticklabels([])
        a[1][idx].set_yticklabels([])
        a[2][idx].set_xticklabels([])
        a[2][idx].set_yticklabels([])
      plt.savefig(FLAGS.figdir+'/'+figname+'.png')

      ## loss L2
      #with open(os.path.join(FLAGS.logdir, ofilename), 'w') as f:
      #  for r in zip(loss_l2_list):
      #    f.write(str(r)+'\n')

      ## Full
      # loss
      with open(os.path.join(FLAGS.logdir, ofilename), 'w') as f:
        for re, ro in zip(loss_reconst_list, loss_rotate_list):
          f.write(str(re)+','+str(ro)+'\n')
      # degree
      with open(os.path.join(FLAGS.logdir, dfilename), 'w') as f:
        f.write("\n".join(" ".join(map(str,x)) for x in (deg_reconst_list, deg_reconst_list)))

      # degree
      #with open(os.path.join(FLAGS.logdir, dfilename), 'w') as f:
      #  for re, ro in zip(deg_reconst_list, deg_rotate_list):
      #    f.writelines(str(re)+','+str(ro)+'\n')

      # rotate-agnostic loss
      #with open(os.path.join(FLAGS.logdir, ofilename), 'w') as f:
      #  for re, re in zip(loss_reconst_list, loss_reconst_list):
      #    f.write(str(re)+','+str(re)+'\n')

      # degree
      #with open(os.path.join(FLAGS.logdir, dfilename), 'w') as f:
      #  f.write("\n".join(" ".join(map(str,x)) for x in deg_reconst_list))

      # bottleneck loss
      #with open(os.path.join(FLAGS.logdir, ofilename), 'w') as f:
      #  for ro, ro in zip(loss_rotate_list, loss_rotate_list):
      #    f.write(str(ro)+','+str(ro)+'\n')


  print("### DEBUG NORMAL END ###")
  # FINISH
  etime = (time.time() -stime)/60.0 # minutes
  print("   Execution time [minutes]  : %f" % etime, flush=True)
