import tensorflow as tf
from osgeo import gdal
import numpy as np


def load_dataset(file_names, side, n_bands):
    """
    Inputs:
        file_names: iterable of file names
        side: length of each side in pixels (images assumed square)
        n_bands: number of spectral bands in the data

    Returns:
        tensorflow iterator over the dataset
    """

    def parser(serialized):
        """Define mask of exported GEE kernels.
        """
        features = {
            f"b{i+1}": tf.FixedLenFeature((side, side), dtype=tf.float32)
            for i in range(n_bands)
        }
        parsed_features = tf.parse_single_example(serialized, features)
        return tf.stack([parsed_features[f"b{i+1}"] for i in range(n_bands)], axis=2)

    return (
        tf.data.TFRecordDataset(tf.constant(file_names))
        .apply(tf.contrib.data.shuffle_and_repeat(1000))
        .map(parser)
        .prefetch(tf.contrib.data.AUTOTUNE)
    )


def full_tiff_pipeline(filenames, shape, batch_size=32):
    """Hopefully boosts performance of tiff file extraction by amortizing
    opening costs. Warning: if the number of bands in `shape` is less than the
    number of bands in the tiff file, then the bands will be randomly_cropped.
    """
    return (
        tf.data.Dataset.from_tensor_slices(filenames)
        .apply(tf.contrib.data.shuffle_and_repeat(10000))
        .map(
            lambda filename: tf.py_func(
                lambda f: gdal.Open(f).ReadAsArray(),
                [filename],
                tf.int16,
                stateful=False,
            )
        )
        .map(
            lambda images: tf.extract_image_patches(
                tf.expand_dims(images, 0),
                ksizes=[1, shape[0], shape[1], 1],
                strides=[1, shape[0] // 2, shape[1] // 2, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
        )
        .interleave(
            lambda imgs: tf.data.Dataset.from_tensor_slices(
                tf.reshape(imgs, [-1, *shape])
            ),
            cycle_length=4,
        )
        .apply(tf.contrib.data.shuffle_and_repeat(10000))
        .apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        .prefetch(1)
    )


def crop_fn(crop_height, crop_width):
    """Returns a function that randomly crops an input image repeatedly. Number
    of output crops depends on how big the image is relative to the crop size.
    """

    def output_fn(full_tiff, height, width, bands):
        count = tf.constant(0)
        crop_list = tf.Variable([], shape=[None, crop_height, crop_width, None])

        # As many random crops as you can fit nicely grid aligned crops
        max_crops = height * width / crop_height / crop_width

        def condition(count, _):
            return count < max_crops

        def append_random_crop(_, crops):
            crop = tf.random_crop(full_tiff, [crop_height, crop_width, bands])
            crops = tf.concat([crops, [crop]], 0)
            return count + 1, crops

        index, crop_list = tf.while_loop(
            condition,
            append_random_crop,
            [count, crop_list],
            shape_invariants=[
                count.get_shape(),
                tf.TensorShape([None, crop_height, crop_width, None]),
            ],
        )
        return tf.data.Dataset.from_tensor_slices(crop_list)

    return output_fn


def read_tiff_gen(tiff_files, side):
    """Returns an initializable generator that reads (side, side, bands) squares
    in the tiff files.
    """

    def gen():
        for f in tiff_files:
            data = gdal.Open(f)
            rows = data.RasterXSize
            cols = data.RasterYSize

            rows -= rows % side
            cols -= cols % side

            for xoff in range(0, rows, side):
                for yoff in range(0, cols, side):
                    bands = []
                    for b in range(data.RasterCount):
                        band = data.GetRasterBand(b + 1).ReadAsArray(
                            xoff=xoff, yoff=yoff, win_xsize=side, win_ysize=side
                        )
                        bands.append(band)

                    assert all(band is not None for band in bands), "index err"

                    img = np.stack(bands, axis=-1)
                    if (img != 0).any():
                        yield img.astype(np.float32)

    return gen


def parse_tiff_fn(img_shape):
    img_width, img_height, n_bands = img_shape

    def parser(serialized):
        features = {
            f"b{i+1}": tf.FixedLenFeature((img_width, img_height), tf.float32)
            for i in range(n_bands)
        }
        x = tf.parse_single_example(serialized, features)
        return tf.stack([x[f"b{i+1}"] for i in range(n_bands)], axis=2)

    return parser


def parse_tfr_fn(img_shape):
    img_width, img_height, n_bands = img_shape

    def parser(serialized):
        features = {
            "rows": tf.FixedLenFeature([], tf.int64),
            "cols": tf.FixedLenFeature([], tf.int64),
            "bands": tf.FixedLenFeature([], tf.int64),
            "vals": tf.FixedLenFeature((img_width, img_height, n_bands), tf.float32),
        }
        x = tf.parse_single_example(serialized, features)
        return x["vals"]

    return parser
