
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential


def residual_add(x, r, data_format='channels_last'):
    """Adds `r` to `x` reshaping `r` and zero_padding extra dimensions
    If the shape doesn't fit, the input tensor is bilinear resized
    If the input tensor has too many channels only the first channels will be added
    If the input tensor has too few channels it will be added to the first output chnanels
    """

    def fn(args):
        # NOTE need to reimport to successfully reload models
        import tensorflow as tf
        from functools import partial

        x, r = args
        x_shape, r_shape = tf.shape(x), tf.shape(r)

        if data_format == "channels_last":
            r_c, x_c = r_shape[3], x_shape[3]
            r_hw, x_hw = r_shape[1:3], x_shape[1:3]

            zero_pad_channel = lambda r: tf.pad(
                r, [[0, 0], [0, 0], [0, 0], [0, x_c - r_c]]
            )
            slice_channel = lambda r: r[:, :, :, :x_c]
            resize = lambda r: tf.image.resize(r, x_hw, method='nearest')

        elif data_format == "channels_last":
            r_c, x_c = r_shape[1], x_shape[1]
            r_hw, x_hw = r_shape[2:4], x_shape[2:4]

            def zero_pad_channel(r):
                return tf.pad(r, [[0, 0], [0, x_c - r_c], [0, 0], [0, 0]])

            slice_channel = lambda r: r[:, :x_c]

            def resize(r):
                # HACK this is certainly not efficient
                r = tf.transpose(r, [0, 3, 1, 2])
                r = tf.image.resize(r, x_hw, method='nearest')
                return tf.transpose(r, [0, 3, 1, 2])

        else:
            raise ValueError("Invalid data_format: `%s`" % data_format)

        r = tf.cond(tf.greater(r_c, x_c), partial(slice_channel, r), lambda: r)
        r = tf.cond(tf.less(r_c, x_c), partial(zero_pad_channel, r), lambda: r)
        r = tf.cond(
            tf.reduce_any(tf.not_equal(r_hw, x_hw)), partial(resize, r), lambda: r
        )

        return x + r

    return Lambda(fn)([x, r])


def resblock(x, filters, nonlinearity, data_format='channels_last'):
    """Two 3x3 convoluions with a residual addition between them
    """
    r = x
    x = nonlinearity()(x)
    x = Conv2D(filters, 3, padding="same", data_format=data_format)(x)
    x = nonlinearity()(x)
    x = Conv2D(filters, 3, padding="same", data_format=data_format)(x)
    return residual_add(x, r, data_format)


def resblocks(x, filters, blocks, nonlinearity, data_format, cardinality=1):
    """Repeatedly applies resblock, supports ResNeXT caridnality
    would be faster if there were grouped convolution ops in tf.
    """
    if cardinality == 1:
        for _ in range(blocks):
            x = resblock(x, filters, nonlinearity, data_format)
    else:
        lanes = []
        for _ in range(cardinality):
            b = x
            for _ in range(blocks):
                b = resblock(b, filters, nonlinearity, data_format)
            lanes.append(b)
        x = Add()(lanes)

    return x

def scale_change_block(
      x, filters, nonlinearity, down, scale=2, data_format="channels_last"
    ):
    """2 3x3 convs with nonlinearities, first one is strided (and ConvT if not down).
    """
    r = x
    x = nonlinearity()(x)
    _conv = Conv2D if down else Conv2DTranspose
    x = _conv(filters, max(3, scale), scale, padding="same", data_format=data_format)(x)
    x = nonlinearity()(x)
    x = Conv2D(filters, 3, padding="same", data_format=data_format)(x)
    return residual_add(x, r, data_format)

def model_resnet_fn(shape=(128,128,6), nblocks=5, base_dim=4, nstack_layer=3,
                    nonlinearity=LeakyReLU, block_len=0,scale=2,
                    data_format="channels_last",):
    """
      block_len: number of recursives in residual 2-conv layer
    """
  
    # set params
    params = {
      'filters': [ 2**(i+base_dim) for i in range(nblocks)],
      'kernel_size': 3
    }  # remainded n-1 blocks

    #----------------------------------------------------------------------------------------------
    # Encoder
    #----------------------------------------------------------------------------------------------
    x = encoder_input = Input(shape=shape, name="encoder_input")
    x = Conv2D(filters=params["filters"][0], kernel_size=3, data_format=data_format,
               padding='same', kernel_initializer='he_normal')(x)
    x = resblocks(x, params["filters"][0], block_len, nonlinearity, data_format)
    for i, filters in enumerate(params["filters"]):
            # Half image size
            x = scale_change_block(
                x, filters, nonlinearity, down=True, scale=scale, data_format=data_format
            )
            x = resblocks(x, filters, block_len, nonlinearity, data_format=data_format)
            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
            x = BatchNormalization()(x)
    x = nonlinearity()(x)
    encoder = Model(encoder_input, x)
    encoded = x

    #----------------------------------------------------------------------------------------------
    # Decoder
    #----------------------------------------------------------------------------------------------
    x = decoder_input = Input(x.shape[1:], name="decoder_input")

    for i, filters in enumerate(reversed(params["filters"])):
            # Double Image size
            x = scale_change_block(
                x, filters, nonlinearity, down=False, scale=scale, data_format=data_format
            )
            x = resblocks(x, filters, block_len, nonlinearity, data_format=data_format)
            x = BatchNormalization()(x)

    input_channels = shape[0] if data_format == "channels_first" else shape[2]
    x = nonlinearity()(x)
    x = Conv2D(input_channels, 1, name="reconstructed", data_format=data_format)(x)
    x = nonlinearity()(x)
    decoder = Model(decoder_input, x, name="decoder")

    #----------------------------------------------------------------------------------------------
    # Encoder + Decoder
    #----------------------------------------------------------------------------------------------

    x = encoded
    for i, filters in enumerate(reversed(params["filters"])):
            # Double Image size
            x = scale_change_block(
                x, filters, nonlinearity, down=False, scale=scale, data_format=data_format
            )
            x = resblocks(x, filters, block_len, nonlinearity, data_format=data_format)
            x = BatchNormalization()(x)

    input_channels = shape[0] if data_format == "channels_first" else shape[2]
    x = nonlinearity()(x)
    x = Conv2D(input_channels, 1, name="reconstructed", data_format=data_format)(x)
    x = nonlinearity()(x)

    # Autoencoder
    autoencoder = Model(encoder_input, x, name="autoencoder")

    return encoder, decoder, autoencoder

