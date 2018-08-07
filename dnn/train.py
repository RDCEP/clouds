import argparse
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import timeline
import pipeline
import model
import os

# Parse Arguments
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("data", help="/path/to/data.tiff")
p.add_argument(
    "model_dir",
    help="/path/to/model/ to load and train or to save new model",
    default=None,
)
p.add_argument("-b", "--batch_size", type=int, help=" ", default=32)
p.add_argument("-o", "--optimizer", help="gradient descent optimizer", default="adam")
p.add_argument("-l", "--loss", help=" ", default="mean_squared_error")
p.add_argument(
    "-e", "--epochs", type=int, help="number of epochs (1000 batches)", default=10
)
p.add_argument(
    "-sh",
    "--shape",
    nargs=3,
    type=int,
    help="Shape of input image",
    default=(64, 64, 7),
)

FLAGS = p.parse_args()
if FLAGS.model_dir[-1] != "/":
    FLAGS.model_dir += "/"


# Load Data
img_width, img_height, n_bands = FLAGS.shape
del img_height  # Unused TODO maybe use it?

data = tf.data.Dataset.from_generator(
    pipeline.read_tiff_gen([FLAGS.data], img_width),
    tf.float32,
    (img_width, img_width, n_bands),
).apply(tf.contrib.data.shuffle_and_repeat(100))


# Load or Define Model
if not os.path.isdir(FLAGS.model_dir):
    os.mkdir(FLAGS.model_dir)

if os.path.exists(FLAGS.model_dir + "model.h5"):
    print("Loading existing model")
    ae = keras.models.load_model(FLAGS.model_dir + "model.h5")
else:
    print("Defining new model")
    _, ae = model.autoencoder(FLAGS.shape)

# Saving
ckpt = keras.callbacks.ModelCheckpoint(FLAGS.model_dir + "model.h5")
# Profiling
# Pending https://github.com/tensorflow/tensorflow/issues/19911
#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata = tf.RunMetadata()

# Compile and train
ae.compile(
    FLAGS.optimizer,
    loss=FLAGS.loss,
    metrics=["mae"],
    #options=run_options,
    #run_metadata=run_metadata,
)
ae.fit(
    x=data.zip((data, data)).batch(FLAGS.batch_size),
    steps_per_epoch=1000,
    epochs=FLAGS.epochs,
    verbose=2,
    callbacks=[ckpt],
)
# Save profiling information
trace = timeline.Timeline(step_stats=run_metadata.step_stats)
with open(FLAGS.model_dir + "timeline.ctf.json", "w") as f:
    f.write(trace.generate_chrome_trace_format())
