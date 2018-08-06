import argparse
import tensorflow as tf
import tensorflow.keras as keras
import pipeline


parser = argparse.ArgumentParser()

parser.add_argument(
    short="-sh",
    long="--shape"
    help="Shape of input image"
    default=(64, 64, 7)
)

parser.add_argument(
    short="-d",
    long="--data"
    help="/path/to/data.tiff"
)
parser.add_argument(
    short="-ct",
    long="--continue_training",
    help="/path/to/model json and h5 file to continue training from",
    default=None
)

flags = parser.parse_args()

# Load Data
img_width, img_height, n_bands = flags.shape
del img_height # Unused TODO maybe use it?

data = (tf.data.Dataset
        .from_generator(
            pipeline.read_tiff_gen([flags.data], img_width),
            tf.float32,
            (img_width, img_width, n_bands)
        )
        .apply(tf.contrib.data.shuffle_and_repeat(100))
       )


# Load or Define Model

if flags.continue_training:
    path = flags.continue_training
    with open(path + ".json", "r") as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights(path + ".h5")

else:
    model = autoencoder(shape)



parser.parse_args()



data = (tf.data.Dataset
        .from_generator(
            pipeline.read_tiff_gen(tiff_files, img_width),
            tf.float32,
            (img_width, img_width, n_bands)
        )
        .apply(tf.contrib.data.shuffle_and_repeat(100))
       )



model.save_weights("model.h5")

loaded_model.load_weights("model.h5")
