import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model


def autoencoder(shape, n_layers=3):
    """
    Returns an encoder model and autoencoder model
    """
    x = inp = Input(shape=shape, name="ae_input")

    # Encoder
    for i in range(n_layers):
        depth = 16 * 2 ** i
        x = Conv2D(depth, 3, activation="relu", padding="same")(x)
        x = MaxPool2D(2, padding="same")(x)

    encoded = x

    # Decoder
    for i in range(n_layers):
        depth = 16 * 2 ** (n_layers - i - 1)
        x = Conv2D(depth, 3, activation="relu", padding="same")(x)
        x = UpSampling2D(2)(x)

    ## Dumb hack to get numbers to work out
    # x = Conv2D(depth, 3, activation='relu')(x)
    # x = Conv2D(depth, 3, activation='relu')(x)

    decoded = Conv2D(shape[-1], 3, activation="relu", padding="same")(x)

    return Model(inp, encoded), Model(inp, decoded)
