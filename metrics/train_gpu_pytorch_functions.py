import os
import gc
import json
import time
import math
import argparse
import itertools
import numpy as np
import pytorch as torch
from datetime import datetime

#TF things: 
#from tensorflow.python.keras.layers import *
#from tensorflow.python.keras.models import Model, Sequential
#from tensorflow.python.client import timeline
#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.profiler import ProfileOptionBuilder, Profiler

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
        sq = _imgs[i] - _imgs[j]
        _loss_rotate_list.append(
          np.mean( torch.mul(sq,sq) ) 
        )
      loss_rotate_list.append(torch.max(_loss_rotate_list))

    loss_rotate = np.mean(torch.stack(loss_rotate_list))

    etime = datetime.now()
    print(" Loss Rotate {} s".format(etime - stime))
    return torch.tensor(c_lambda ,dtype=torch.float32) * loss_rotate  

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
        rimgs = imgs.get_rotation_matrix2d(angle=torch.tensor(angle ,dtype=torch.float32))
            
        return rimgs

    loss_reconst_list = []
    angle_list = [i*math.pi/180 for i in range(1,360,dangle)]
     
    # 08/28 2PM  before modification 
    encoded_imgs = encoder(oimgs)
    reconst_imgs = decoder(encoded_imgs)
    for angle in angle_list:
      rimgs = rotate_operation(reconst_imgs,angle=angle) # R_theta(x_hat)
      sq = imgs - rimgs
      loss_reconst_list.append(np.mean(torch.mul(sq,sq))) 
    loss_reconst = torch.min(loss_reconst_list)
    etime = datetime.now()
    print(" Loss Reconst {} s".format(etime - stime))
    return loss_reconst, loss_reconst_list

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
    img_list.extend([np.reshape(tmp_img_tf, (1,28,28,1))] )
    img_list.extend([ (tmp_img_tf.clone()).unsqueeze_(0) for i in range(copy_size-1)])

  coimgs = torch.cat(img_list, dim=0)
  crimgs = rotate_fn(coimgs, seed=np.random.randint(0,999), return_np=False)
  etime = datetime.now()
  print(" make_copy_rotate {} s".format(etime - stime))
  return crimgs, coimgs

