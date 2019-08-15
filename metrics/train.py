
import matplotlib
matplotlib.use('Agg')

import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.examples.tutorials.mnist import input_data

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
    '--num_epoch',
    type=int,
    default=5
  )
  p.add_argument(
    '--batch_size',
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
    x = Conv2D(filters=16, kernel_size=3, padding='same')(x)
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

def loss_dev_fn(output_layer, 
                input_layer, 
                encoded_imgs,
                encoder,
                batch_size=32, dangle=2, c_lambda=1
               ):
    
    def rotate_opetation(imgs, angle=1):
        rimgs = tf.contrib.image.rotate(
                imgs,
                tf.constant(angle ,dtype=tf.float32),
                interpolation='NEAREST',
                name=None
        )
        return rimgs
    
    # loss lists
    loss_reconst = [] # first term
    loss_hidden  = [] # seconds term
    
    angle_list = [i for i in range(1,360,dangle)]
    for angle in angle_list:
        rimgs = rotate_opetation(output_layer) # R_theta(x_hat)
        rencoded_imgs = rotate_opetation(rotate_opetation(input_layer)) # Z(R(x))
        
        # loss
        loss_reconst.append(tf.reduce_mean(tf.square(input_layer - rimgs)) )
        loss_hidden.append(tf.reduce_mean(tf.square(encoded_imgs - rencoded_imgs)))
    
    # Get min-max
    reconst = tf.reduce_min(tf.stack(loss_reconst))
    hidden  = tf.reduce_max(tf.stack(loss_reconst))
    
    return reconst + tf.multiply(tf.constant(c_lambda ,dtype=tf.float32), hidden)

def input_fn(data, batch_size=32):
    data1 = data.reshape(-1,28,28,1)
    dataset = tf.data.Dataset.from_tensor_slices((data1))
    #dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    dataset = dataset.repeat().batch(batch_size)
    return dataset


if __name__ == "__main__":
  # set data
  mnist = input_data.read_data_sets(os.path.abspath("./MNIST_data/"), one_hot=True)

  # get arg-parse as FLAGS
  FLAGS = get_args()
  # make dirs
  os.makedirs(FLAGS.logdir, exist_ok=True)
  os.makedirs(FLAGS.figdir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir, exist_ok=True)

  # ad-hoc params
  num_test_images = 10

  # get model
  encoder, decoder = model_fn()

  # get dataset and one-shot-iterator
  dataset = input_fn(mnist.train.images, batch_size=FLAGS.batch_size)
  img= dataset.make_one_shot_iterator().get_next()

  # get layer output
  encoder_img = encoder(img)
  decoder_img = decoder(encoder_img)

  # compute loss and train_ops
  loss = loss_dev_fn(decoder_img, img, encoder_img,
                   encoder,
                   batch_size=FLAGS.batch_size, dangle=FLAGS.dangle, c_lambda=FLAGS.c_lambda)
  train_ops = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

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
  bname1 = 'new_nepoch-'+str(FLAGS.num_epoch)+'_lr-'+str(FLAGS.lr)
  bname2 = '_nbatch-'+str(FLAGS.batch_size)+'_lambda'+str(FLAGS.c_lambda)+'_dangle'+str(FLAGS.dangle)
  figname   = 'fig_'+bname1+bname2
  ofilename = 'loss_'+bname1+bname2+'.txt'

  # start!
  stime = time.time()
  with tf.Session() as sess:
    sess.run(init)
    for epoch in range(FLAGS.num_epoch):
        num_batches=mnist.train.num_examples//FLAGS.batch_size
        for iteration in range(num_batches):
            X_batch,y_batch=mnist.train.next_batch(FLAGS.batch_size)
            sess.run(train_ops)
        
        train_loss=loss.eval(feed_dict={X:X_batch.reshape(-1,28,28,1)})
        print("epoch {} loss {}".format(epoch,train_loss), flush=True)   
        # save for checkio
        train_loss_list.append(train_loss)
    
        # save model at every N steps
        if epoch % FLAGS.save_every == 0:
          for m in save_models:
            save_models[m].save_weights(
              os.path.join(
                FLAGS.output_modeldir, "{}-{}.h5".format(m, epoch)
              )
            )

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
