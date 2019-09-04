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
    #_x = Flatten()(x)
    #_x = Dense(10)(_x) # off this layer in general. Only for sanity check
    #encoder = Model(inp, _x, name='encoder')
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

def resize_image_fn(imgs,labels, height=32, width=32):
  """Resize images from (#batch,H',W',C) --> (#batch,H,W,C)
  """
  reimgs = tf.image.resize_images(imgs, (height, width))
  return reimgs, labels

def input_fn(data,labels, batch_size=32):
    data   = np.reshape(data, (-1,28,28,1))
    #labels = labels.astype(np.float32)+1.0e-12
    # ++ common version
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.map(resize_image_fn)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

def loss_fn(imgs, encoder, decoder):
    rimgs = decoder(encoder(imgs))
    return tf.reduce_mean(tf.square(imgs - rimgs))

    #       )

if __name__ == '__main__':
  # set data
  mnist = input_data.read_data_sets(os.path.abspath("./MNIST_data/"), one_hot=True)

  # get arg-parse as FLAGS
  FLAGS = get_args()

  # diretory
  os.makedirs(FLAGS.output_modeldir, exist_ok=True)

  # get dataset and one-shot-iterator
  dataset = input_fn(mnist.train.images,mnist.train.labels, 
                     batch_size=FLAGS.batch_size)
  # why get_one_shot_iterator leads OOM error?
  train_iterator = dataset.make_initializable_iterator()
  imgs, labels = train_iterator.get_next()

  # get model
  encoder, decoder = model_fn(nblocks=FLAGS.nblocks)
  print("\n {} \n".format(encoder.summary()), flush=True)
  print("\n {} \n".format(decoder.summary()), flush=True)

  # loss + optimizer
  loss = loss_fn(imgs,encoder,decoder)
  train_ops = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

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
  
  # TRAINING
  with tf.Session(config=config) as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # initial run
    init=tf.global_variables_initializer()
    X=tf.placeholder(tf.float32,shape=[None,28,28,1])
    sess.run(init)
    sess.run(train_iterator.initializer)

    # initialize other variables
    num_batches=mnist.train.num_examples//FLAGS.batch_size

    # Trace and Profiling options
    run_metadata = tf.RunMetadata()
    run_opts = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)

    stime = time.time()
    for epoch in range(FLAGS.num_epoch):
        for iteration in range(num_batches):
          _, train_loss = sess.run([train_ops, loss])

          # check
          if iteration % 20 == 0:
             print(" Iteration {} Loss {}".format(iteration, train_loss))

        X_batch,y_batch=mnist.train.next_batch(FLAGS.batch_size)
        train_loss = loss.eval(feed_dict={X:X_batch.reshape(-1,28,28,1)})
        print("\n Epoch {}  Loss  {}\n".format( epoch, train_loss), flush=True)   
    
        # save model at every N steps
        if epoch % FLAGS.save_every == 0:
          for m in save_models:
            save_models[m].save_weights(
              os.path.join(
                FLAGS.output_modeldir, "{}-{}.h5".format(m, epoch)
              )
            )
        
  # FINISH
  etime = (time.time() -stime)/60.0 # minutes
  print("### NORMAL END ###")
  print("   Execution time [minutes]  : %f" % etime, flush=True)
