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
    #encoder = Model(inp, x, name='encoder')
    encoder = x
    return encoder , inp

def model_tf_fn(encoder, inp):
    fc_layer = Flatten()(encoder)
    x = Dense(128, activation='relu')(fc_layer)
    x = Dense(10, activation='softmax')(x)
    tf_model = Model(inp, x, name='transfer')
    return tf_model

def load_weight(model_dir='.', epoch=10):
    encoder_def = model_dir+'/encoder.json'
    encoder_weight = model_dir+'/encoder-'+str(epoch)+'.h5'
    with open(encoder_def, "r") as f:
        encoder = tf.keras.models.model_from_json(f.read())
    encoder.load_weights(encoder_weight)
    return encoder

def resize_image_fn(imgs,height=32, width=32):
  """Resize images from (#batch,H',W',C) --> (#batch,H,W,C)
  """
  reimgs = tf.image.resize_images(imgs, (height, width))
  return reimgs

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
  os.makedirs(FLAGS.output_modeldir, exist_ok=True)

  # get model
  encoder, inp  = model_fn(nblocks=FLAGS.nblocks) # not Modelized
  classifier = model_tf_fn(encoder, inp) # get model
  print("\n {} \n".format(classifier.summary()), flush=True)

  # load trained weight
  trained_encoder = load_weight(model_dir=FLAGS.transfer_modeldir, epoch=FLAGS.tf_epoch)

  # weight
  nlayer = 0
  for lx, ly in zip(classifier.layers[:-1], trained_encoder.layers):
    lx.set_weights(ly.get_weights())
    nlayer+=1

  # Fix weight 
  for ilayer in classifier.layers[:nlayer]:
    ilayer.trinable = False

  # Compile model
  classifier.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer=tf.keras.optimizers.Adam(FLAGS.lr),
      metrics=['accuracy']
  )

  # dataset
  imgs_tf =  resize_image_fn(mnist.train.images.reshape(-1,28,28,1),
                             height=FLAGS.height, 
                             width=FLAGS.width
  )
  imgs = tf.keras.backend.eval(imgs_tf)
  print(imgs.shape)
  
  # set-up save models
  save_models = {"classifier": classifier}
  # save model definition
  for m in save_models:
    with open(os.path.join(FLAGS.output_modeldir, m+'.json'), 'w') as f:
      f.write(save_models[m].to_json())

  # set save-cehckpoints
  checkpoint_path = FLAGS.output_modeldir
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_weights_only=True,verbose=1
  )

  # gpu config 
  config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True
    ),
    log_device_placement=False
  )
  stime = time.time()

  # TRAINING
  num_batches=int(mnist.train.num_examples//FLAGS.batch_size*0.8)
  # 80% training 20 % validation
  classifier.fit(
    imgs,
    mnist.train.labels,
    batch_size=None,
    steps_per_epoch=num_batches,
    epochs=FLAGS.num_epoch,
    validation_split=0.2,
    verbose=1,
    callbacks = [cp_callback]
  )

  # save models
  for m in save_models:
    save_models[m].save_weights(
      os.path.join(
        FLAGS.output_modeldir, "{}-{}.h5".format(m, FLAGS.num_epoch)
      )
    )
  
        
  # FINISH
  etime = (time.time() -stime)/60.0 # minutes
  print("### NORMAL END ###")
  print("   Execution time [minutes]  : %f" % etime, flush=True)
