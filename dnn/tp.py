import tensorflow as tf
from osgeo import gdal
import pipeline

import time


files = [
    "data/tif/2017-01-01_MOD09GA_background_removal_zero_inputated_image_with_cf_50perc_grid_size10-0000017664-0000000000.tif"
]

t0 = time.time()

data = pipeline.full_tiff_pipeline(files, (64, 64, 7))

print(f"Time to initialize dataset {time.time() - t0}")

t0 = time.time()

x = data.make_one_shot_iterator().get_next()
print(f"Time to get next {time.time() - t0}")

with tf.Session() as sess:
    t0 = time.time()
    y = sess.run(x)
    print(f"Time to run first {time.time() - t0}")
    # print(y)

    t0 = time.time()
    sess.run(x)
    print(f"Time to run second {time.time() - t0}")

    t0 = time.time()
    for i in range(100):
        sess.run(x)
    print(f"Time to run third to 103rd {time.time() - t0}")
