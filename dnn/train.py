import argparse
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import timeline
import pipeline
import model
import subprocess
from os import path, mkdir

# Parse Arguments
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("data_glob", help="pattern to pick up tf records files")
p.add_argument(
    "model_dir",
    help="/path/to/model/ to load and train or to save new model",
    default=None,
)
p.add_argument("-b", "--batch_size", type=int, help=" ", default=32)
p.add_argument("-o", "--optimizer", help="gradient descent optimizer", default="adam")
p.add_argument("-l", "--loss", help=" ", default="mean_squared_error")
p.add_argument("-spe", "--steps_per_epoch", help=" ", type=int, default=1000)
p.add_argument(
    "-e", "--epochs", type=int, help="number of epochs (1000 batches)", default=10
)
p.add_argument("-nm", "--new_model", help="Name of model in model.py to use")
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

# Log meta data
commit = subprocess.check_output(["git", "describe", "--always"]).strip()
print(f"Tensorflow version: {tf.__version__}")
print(f"Current Git Commit: {commit}")
print("Flags:")
for f in FLAGS.__dict__:
    print(f"\t{f}:{(20-len(f)) * ' '} {FLAGS.__dict__[f]}")


# Load Data
img_width, img_height, n_bands = FLAGS.shape

features = {
    f"b{i+1}": tf.FixedLenFeature((img_width, img_height), tf.float32)
    for i in range(n_bands)
}


def stack_bands(x):
    return tf.stack([x[f"b{i+1}"] for i in range(n_bands)], axis=2)


data = (
    tf.data.Dataset.list_files(FLAGS.data_glob)
    .flat_map(tf.data.TFRecordDataset)
    .map(lambda serialized: tf.parse_single_example(serialized, features))
    .map(stack_bands)
    .batch(FLAGS.batch_size)
)

# Load or Define Model
if not path.isdir(FLAGS.model_dir):
    mkdir(FLAGS.model_dir)

model_file = path.join(FLAGS.model_dir, "model.h5")

if path.exists(model_file):
    print(f"Loading existing model")
    ae = keras.models.load_model(model_file)
else:
    if FLAGS.new_model:
        build_model = getattr(model, FLAGS.new_model)
    else:
        build_model = model.autoencoder

    print(f"Defining new model from {build_model}")
    _, ae = build_model(FLAGS.shape)

ae.summary()

# Saving
ckpt = keras.callbacks.ModelCheckpoint(model_file)
# # Profiling
# # Pending https://github.com/tensorflow/tensorflow/issues/19911
# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()
# Tensorboard
tensorboard = keras.callbacks.TensorBoard(
    log_dir=path.join(FLAGS.model_dir, "/tb_logs"),
    histogram_freq=0,
    batch_size=FLAGS.batch_size,
    write_graph=True,
    write_grads=False,
    write_images=True,
)

# Compile and train
ae.compile(
    FLAGS.optimizer,
    loss=FLAGS.loss,
    metrics=["mae", "mse"],
    # options=run_options,
    # run_metadata=run_metadata,
)
ae.fit(
    x=data.zip((data, data)),
    steps_per_epoch=FLAGS.steps_per_epoch,
    epochs=FLAGS.epochs,
    verbose=2,
    callbacks=[ckpt, tensorboard],
)
# Save profiling information
# trace = timeline.Timeline(step_stats=run_metadata.step_stats)
# with open(FLAGS.model_dir + "timeline.ctf.json", "w") as f:
#     f.write(trace.generate_chrome_trace_format())
