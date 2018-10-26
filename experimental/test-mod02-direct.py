import tensorflow as tf
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from reproduction.pipeline.load import load_data_direct
from reproduction.pipeline.global_norm import MOD02_FIELDS
from matplotlib import pyplot as plt

ds = load_data_direct(
    "data/mod02-1km/MOD021KM.A2017001.2330.006.2017004135453.hdf",
    [128, 128, 38],
    MOD02_FIELDS,
    "j",
    shuffle_buffer_size=10,
)

x = ds.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n, c, imgs = sess.run(x)

rows, _, _, cols = imgs.shape

fig, ax = plt.subplots(rows + 1, cols, figsize=(cols * 2, rows * 2))

for i in range(rows):
    for j in range(cols):
        a = ax[i, j]
        a.imshow(imgs[i, :, :, j], cmap="bone")
        a.set_xticks([])
        a.set_yticks([])

for j in range(cols):
    ax[rows, j].hist(imgs[:, :, :, j].ravel())

fig.tight_layout()
fig.savefig("foobar.png")
