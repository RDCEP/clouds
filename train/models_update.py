
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential

def model_synmetric_resize_fn(shape=(128,128,6), 
                              nblocks=5, base_dim=4, nstack_layer=3,
                              rheight=32, rwidth=32) :
  """
      Add resize operation as a layer for first layer at the encoder 
      and the last layer at the decoder.
      Synmetrick before and after final block for encoder 
      Reference: https://blog.keras.io/building-autoencoders-in-keras.html
      base_dim: 4
      nblocks : 5
  """
  def convSeries_fn(x,
                    filters=16, 
                    kernel_size=3, 
                    nstack_layer=3, 
                    stride=2, 
                    init_filter=16,
                    channels=6,
                    up=True, 
                    ):
    """
    INPUT
      nstack_layer : number of iteration of conv layer before batch_norm. default 3.
      up           : boolean. True is encoder, False is decoder(conv2D transpose)
    """
    if up:
      for idx in range(nstack_layer-1):
          x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                   strides=1, kernel_initializer='he_normal')(x)
          x = LeakyReLU()(x)
      return x
    else:
      for idx in range(nstack_layer):
        if idx == 0:
          x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, 
                              strides=(stride,stride), padding='same')(x)
        else:
          x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                   kernel_initializer='he_normal')(x)
          if idx == nstack_layer - 1:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
      return x

  def encoder_block(x, filters=32, kernel_size=3, stride=2):
    """ Stride should be 2 for downsampling
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
        strides=stride, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

  def resize_layer(x, height, width):
    """ assume channle is last [#batch, h, w, c] """
    def _resize_fn(x):
      import tensorflow as tf
      r_x = tf.image.resize(x, [height, width], method='nearest')
      return r_x
    return Lambda(_resize_fn)(x)
    

  # set params
  params = {
    'filters': [ 2**(i+base_dim) for i in range(nblocks)],
    'kernel_size': 3
  }  # remainded n-1 blocks
  channels = shape[-1]

  #----------------------------------------------------------------------------------------------
  # encoder layers
  #----------------------------------------------------------------------------------------------
  ## start construction
  x = encoder_input = Input(shape=shape, name='encoding_input')
  oheight, owidth, _ = shape
  x = resize_layer(x, rheight, rwidth)
  x = Conv2D(filters=params["filters"][0], kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
  x = LeakyReLU()(x)
  for iblock in range(nblocks):
    filters = params["filters"][iblock]
    kernel_size = params["kernel_size"]
    x = convSeries_fn(x,filters=filters, kernel_size=kernel_size, nstack_layer=nstack_layer, up=True)


    # DownSample
    if iblock < nblocks-1:
      filters = params["filters"][iblock+1]
      x = encoder_block(x, filters=filters, kernel_size=kernel_size)
           
  # build model for encoder + digit layer
  encoder = Model(encoder_input, x, name='encoder')
           
  #----------------------------------------------------------------------------------------------
  # decoder layers
  #----------------------------------------------------------------------------------------------
  x = decoder_input = Input(x.shape[1:], name="decoder_input")
  for iblock in range(nblocks-1):
    filters = params["filters"][::-1][iblock+1]
    kernel_size = params["kernel_size"]

    if iblock == nblocks-1:
        x = convSeries_fn(x,filters=filters, kernel_size=kernel_size,
            nstack_layer=nstack_layer, init_filter=params['filters'][0],channels=channels, up=False)
    else:
        x = convSeries_fn(x,filters=filters, kernel_size=kernel_size,
            nstack_layer=nstack_layer, init_filter=params['filters'][0],channels=channels, up=False)
 
  x = Conv2D(filters=channels, kernel_size=params["kernel_size"], padding='same', kernel_initializer='he_normal')(x)
  x = LeakyReLU()(x)
  # apply resize operation
  x = resize_layer(x, oheight, owidth)
  decoder = Model(decoder_input, x, name='decoder')
           
  return encoder, decoder
  


    

    
