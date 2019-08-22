import os
import json
import math
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
    '--model_dir',
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
    '--save_every',
    type=int,
    default=10
  )
  p.add_argument(
    "--shape",
    nargs=3,
    type=int,
    metavar=("h", "w", "c"),
    help="Shape of input image",
    default=(7, 7, 10),
  )
  p.add_argument(
    '--depth',
    type=int,
    default=10
  )
  p.add_argument(
    '--step',
    type=int,
    default=5
  )
  p.add_argument(
    '--rotation',
    action="store_true", 
    default=False
  )

  args = p.parse_args()
  for f in args.__dict__:
    print("\t", f, (25 - len(f)) * " ", args.__dict__[f])
  print("\n")
  return args


def model_fn(depth=10, shape=(7,7,10)):
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=depth, kernel_size=2, padding='same', activation='relu', input_shape=shape)) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=int(depth/2), kernel_size=2, padding='same', activation='relu'))
    #model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(18, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

def load_model(model_dir='.', step=30):
    encoder_def = model_dir+'/encoder.json'
    encoder_weight = model_dir+'/encoder-'+str(step)+'.h5'
    with open(encoder_def, "r") as f:
        encoder = tf.keras.models.model_from_json(f.read())
    encoder.load_weights(encoder_weight)
    return encoder

def rotate_fn(images):
    """
    Apply random rotation to data and parse to dataset module
    images: before parse to encoder
    """
    # float point 32
    images = images.astype(np.float32)

    # random rotation
    random_angles = tf.random.uniform(
        shape = (tf.shape(images)[0], ), 
        minval = 0*math.pi/180,
        maxval = 359*math.pi/180,
        dtype=tf.float32,
        seed = 0
    )
    rotated_tensor_images = tf.contrib.image.transform(
      images,
      tf.contrib.image.angles_to_projective_transforms(
        random_angles, tf.cast(tf.shape(images)[1], tf.float32), 
            tf.cast(tf.shape(images)[2], tf.float32)
        )
    )
    
    # convert from tensor to numpy
    sess = tf.Session()
    with sess.as_default():
      rotated_images = rotated_tensor_images.eval()
    return rotated_images


def input_fn(data, label, batch_size=32, rotate=True):
    # convert data to model
    train_data = mnist.train.images
    n = train_data.shape[0]
    train_data = train_data.reshape(n,28,28,1)

    if rotate:
      # apply rotation
      train_data = rotate_fn(train_data)

    # load unsupervised model
    encoder = load_model(model_dir=FLAGS.model_dir,step=FLAGS.step) 
    x_train = encoder.predict(train_data) 

    dataset = tf.data.Dataset.from_tensor_slices((x_train, label))

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


if __name__ == "__main__":
  # set data
  mnist = input_data.read_data_sets(os.path.abspath("./MNIST_data/"), one_hot=True)

  # get arg-parse as FLAGS
  FLAGS = get_args()
  # make dirs
  os.makedirs(FLAGS.logdir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir, exist_ok=True)

  #  from_logits should be False if label is one-hot
  _train_labels = np.asarray(mnist.train.labels, np.float32)
  train_labels = tf.convert_to_tensor(_train_labels, np.float32) 

  # get dataset and one-shot-iterator
  dataset = input_fn(mnist.train.images, train_labels, batch_size=FLAGS.batch_size, rotate=FLAGS.rotation)
  img, labels= dataset.make_one_shot_iterator().get_next()

  #  model
  model = model_fn(depth=FLAGS.depth, shape=FLAGS.shape)
 
  # compute loss
  loss = tf.keras.losses.categorical_crossentropy(
    model(img),
    labels
  )  
    #from_logits=False,
    #label_smoothing=0
  train_ops = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

  # set save model
  save_model = {"cnn": model} 
  # save model definition
  for m in save_model:
    with open(os.path.join(FLAGS.output_modeldir, m+'.json'), 'w') as f:
      f.write(save_model[m].to_json())

  #====================================================================
  # Training
  #====================================================================
  # initialize
  init=tf.global_variables_initializer()
  X=tf.placeholder(tf.float32,shape=[None,10])
  Y=tf.placeholder(tf.float32,shape=[None,10])
  train_loss_list = []
  
  # outputnames
  if FLAGS.rotation:
     rotate='rotate'
  else:
     rotate='no-rotate'
  bname1 = 'nepoch-'+str(FLAGS.num_epoch)+'_lr-'+str(FLAGS.lr)
  bname2 = '_nbatch-'+str(FLAGS.batch_size)+rotate
  ofilename = 'loss_cnn_'+bname1+bname2+'.txt'


  # prep test data with rotation and encoding
  encoder = load_model(model_dir=FLAGS.model_dir,step=FLAGS.step) 
  #test_data = mnist.test.images
  #rotated_test_data = rotate_fn(test_data.reshape(-1,28,28,1))
  #x_test  = encoder.predict(rotated_test_data) 

  # start!
  stime = time.time()
  with tf.Session() as sess:
    sess.run(init)
    for epoch in range(FLAGS.num_epoch):
        num_batches=mnist.train.num_examples//FLAGS.batch_size
        for iteration in range(num_batches):
            sess.run(train_ops)
        
        # get loss
        #x_batch,y_batch=mnist.train.next_batch(FLAGS.batch_size)

        # data
        #x_rotate_batch = rotate_fn(x_batch.reshape(-1,28,28,1))
        #X_batch = encoder.predict(x_rotate_batch)

        # label
        #Y_labels = np.asarray(y_batch, np.float32)
        #Y_batch = tf.convert_to_tensor(_batch_labels, np.float32) 
        X_batch = model(img).eval()
        Y_batch = labels.eval()
        train_loss=loss.eval(feed_dict={X:X_batch, Y:Y_batch})
        print("epoch {} loss {}".format(epoch,train_loss), flush=True)   
        # save for checkio
        train_loss_list.append(train_loss)
    
        # save model at every N steps
        if epoch % FLAGS.save_every == 0:
          for m in save_model:
            save_model[m].save_weights(
              os.path.join(
                FLAGS.output_modeldir, "{}-{}.h5".format(m, epoch)
              )
            )
    # get score
    #score = model.evaluate(x_test, mnist.test.labels, verbose=0)

    #print("")
    #print(" Test Accuracy {} ".format(score[1]), flush=True)
  
  etime = (time.time() -stime)/60.0 # minutes
  print("   Training time [minutes]  : %f" % etime, flush=True)
  print(" ### FINISH TRAINING ### ")

  # save loss result
  with open(os.path.join(FLAGS.logdir, ofilename), 'w') as f:
    for ie in train_loss_list:
      f.write(str(ie)+'\n')

  ##===================================================================
  ## Evaluate the model on test set
  ##===================================================================

  # FINISH
  etime = (time.time() -stime)/60.0 # minutes
  print("")
  print("### NORMAL END ###")
  print("   Execution time [minutes]  : %f" % etime, flush=True)
