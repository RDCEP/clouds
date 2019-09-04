#
# + Transfer trained AE model + Dense layer for classifier
#
import os
import gc
import json
import time
import math
import argparse
import itertools
import numpy as np
import tensorflow as tf
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
    '--transfer_modeldir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--output_modeldir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--tf_epoch',
    type=int,
    default=5
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

def model_tf_fn(encoder, shape=(2,2,64)):
    inp = Input(shape=shape) 
    fc_layer = Flatten(encoder)
    x = Dense(10, activation='relu')(fc_layer)
    tf_model = Model(inp, x, name='transfer')
    return x

def load_model(model_dir='.', epoch=10):
    encoder_def = model_dir+'/encoder.json'
    encoder_weight = model_dir+'/encoder-'+str(epoch)+'.h5'
    with open(encoder_def, "r") as f:
        encoder = tf.keras.models.model_from_json(f.read())
    encoder.load_weights(encoder_weight)
    return encoder

def resize_image_fn(imgs,labels, height=32, width=32):
  """Resize images from (#batch,H',W',C) --> (#batch,H,W,C)
  """
  reimgs = tf.image.resize_images(imgs, (height, width))
  return reimgs, labels

def input_fn(data,labels, batch_size=32):
    data   = np.reshape(data, (-1,28,28,1))
    labels = labels.astype(np.float32)
    # ++ common version
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.map(resize_image_fn)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

def loss_fn(imgs, labels, model):
    preds = model(imgs)
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(labels, preds)
    )
    
if __name__ == "__main__":
  # set data
  mnist = input_data.read_data_sets(os.path.abspath("./MNIST_data/"), one_hot=True)

  # get arg-parse as FLAGS
  FLAGS = get_args()

  # diretory
  #os.makedirs(FLAGS.output_modeldir, exist_ok=True)

  # get dataset and one-shot-iterator
  dataset = input_fn(mnist.train.images,mnist.train.labels, 
                     batch_size=FLAGS.batch_size)
  train_iterator = dataset.make_initializable_iterator()
  imgs, labels = train_iterator.get_next()

  # get model
  encoder = load_model(model_dir=FLAGS.transfer_modeldir, epoch=FLAGS.tf_epoch)
  classifier = model_tf_fn(encoder)
  print("\n {} \n".format(classifier.summary()), flush=True)
  stop  

  # loss + optimizer
  loss = loss_fn(imgs,labels,tf_model)  # loss loss
  train_ops = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

  # set-up save models
  save_models = {"classifier": classifier}

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

    #stime = time.time()
    #for epoch in range(FLAGS.num_epoch):
    #    for iteration in range(num_batches):
