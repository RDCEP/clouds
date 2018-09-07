import tensorflow as tf
from osgeo import gdal, ogr
import numpy as np
import json
from tensorflow.contrib.data import shuffle_and_repeat, batch_and_drop_remainder


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


def read_tiff_gen_withloc(tiff_files, side):
    """Returns an initializable generator that reads (side, side, bands) squares
    in the tiff files. Same version as above, with geolocation of the patches
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
                    centroids = []
                    for b in range(data.RasterCount):
                        band = data.GetRasterBand(b + 1).ReadAsArray(
                            xoff=xoff, yoff=yoff, win_xsize=side, win_ysize=side
                        )
                        # TODO: Implement the centroid extraction. Option would be combining with OGR.
                        # TODO: Option two get using rasterio...
                        # centroids.append(data.Centroid())

                        bands.append(band)

                    assert all(band is not None for band in bands), "index err"

                    img = np.stack(bands, axis=-1)
                    if (img != 0).any():
                        yield img.astype(np.float32)

                    # Get coordinates of patch centroid

                    print(centroids)

    return gen


def main_parser(fields, meta_json, saved_as_bytes=True):
    """Parses tfrecord example with chosen fields.
    Note that the shape is not consistent between fields or even between records
    as such you should use the meta_json associated with the particular record
    or pass a record that has been saved with the meta data and as bytes. This
    will fail if `fields` are not the same shape as they cannot be
    concatenated (FIXME with up/down sampling to align shapes?).

    Arguments:
        fields: Fields to select from record
        meta_json: path to json meta datafile mapping fields to num channels and type
        saved_as_bytes: boolean that indicates the tf record contains shape info
            and is saved as bytes
    Returns:
        chans: The number of channels of the output image, the sum of channels
            in selected fields.
        parser:  Returns a 3d tensor from a serialized tf-record.
    """
    fields.sort()
    with open(meta_json, "r") as f:
        meta = json.load(f)

    type_map = {
        "float32": tf.float32,
        "float64": tf.float64,
        "int8": tf.int8,
        "int16": tf.int16,
    }

    features = {}
    chans = 0
    for field in fields:
        c, ty = meta[field][-2:]
        features[field] = tf.FixedLenFeature([], tf.string)
        features[field + "/shape"] = tf.FixedLenFeature([3], tf.int64)
        chans += c

    def parser(ser):
        record = tf.parse_single_example(ser, features)
        res = []
        for f in fields:
            sh = record[f + "/shape"]
            ty = type_map[meta[f][-1]]
            decoded = tf.decode_raw(record[f], ty)
            decoded = tf.reshape(decoded, sh)
            decoded = tf.cast(decoded, tf.float32)
            res.append(decoded)
        return tf.concat(res, axis=2)

    return chans, parser


# TODO depricate
def patchify_fn(height, width, chans):
    """Breaks up a big image into many half overlaping images.
    """

    def fn(img):
        imgs = tf.extract_image_patches(
            images=tf.expand_dims(img, 0),
            ksizes=[1, height, width, 1],
            strides=[1, height // 2, width // 2, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        imgs = tf.reshape(imgs, [-1, height, width, chans])
        return tf.data.Dataset.from_tensor_slices(imgs)

    return fn


def add_pipeline_cli_arguments(p):
    p.add_argument(
        "--data", help="pattern to pick up tf records files", required=True, nargs="+"
    )
    p.add_argument(
        "--fields",
        nargs="+",
        help=(
            "fields to select from tf record. For tif data, it should be b1..bN "
            "where N is the number of bands. For hdf data, it should be the "
            "names of the fields"
        ),
        required=True,
    )
    p.add_argument(
        "--meta_json",
        help="json file mapping field name to number of channels and type.",
        required=True,
    )
    p.add_argument(
        "--shape", nargs=2, type=int, help="Shape of input image", default=(64, 64)
    )

    p.add_argument("--batch_size", type=int, help=" ", default=32)
    p.add_argument(
        "--normalization",
        choices=["max_scale", "whiten", "mean_sub", "none"],
        default="max_scale",
        help="Method for normalizing swath before extracting patches.",
    )
    p.add_argument("--read_threads", type=int, default=4)
    p.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="Size of prefetch buffers in dataset pipeline",
    )
    p.add_argument("--shuffle_buffer_size", type=int, default=10000)

# TODO depricate once inlined in fused_read_fn
def heterogenous_bands(threshold):
    """Returns True if a band in the image has too much of a single value in
    `threshold` fraction of the image. Presumably this value represents no cloud
    as clouds will be more heterogenous.
    """

    def fn(img):
        has_data = []
        for band in tf.unstack(img, axis=-1):
            _, _, count = tf.unique_with_counts(tf.reshape(band, [-1]))
            has_data.append(tf.reduce_max(count) / tf.size(band) < threshold)
        return tf.reduce_any(has_data)

    return fn

# TODO inline with fused_read_fn / refactor to not curry
def normalizer_fn(normalization):
    """Returns a function that performs `normalization` to an image tensor.

    FIXME: This does not mask pixels with no clouds in the calculation of
    moments, biasing mean and variance, We want to normalize by conditional
    mean and conditional variance (conditional on 'pixel with a cloud').
    """

    def fn(img):
        img = tf.verify_tensor_all_finite(img, "Nan before normalizing")
        img = tf.clip_by_value(img, 0, 1e10)

        if normalization == "mean_sub":
            mean, _ = tf.nn.moments(img, (0, 1))
            img -= mean

        elif normalization == "whiten":
            mean, var = tf.nn.moments(img, (0, 1))
            img = (img - mean) / tf.sqrt(var)

        elif normalization == "max_scale":
            img /= tf.reduce_max(img, (0, 1))

        elif normalization == "none":
            pass

        else:
            raise ValueError(f"Unrecognized normalization choice: `{normalization}`")

        return img

    return fn

# TODO rename -- more descriptive
def fused_read_fn(parser, normalize, shape, n=10, seed=None):
    """Parses tf record, normalizes it, throws away if NaNs, and returns patches.
    This is actually 10x faster than the non fused version.
    """
    def extract_patches(swath):
        height, width, chans = shape
        imgs = tf.extract_image_patches(
            images=tf.expand_dims(swath, 0),
            ksizes=[1, height, width, 1],
            strides=[1, height // 2, width // 2, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        imgs = tf.reshape(imgs, [-1, height, width, chans])
        # TODO filter homogenous patches away here
        imgs = tf.map_fn(
            lambda x: tf.image.random_flip_up_down(tf.image.random_flip_left_right(x)),
            imgs
        )
        return imgs

    def fn(ser):
        swath = parser(ser)
        swath = tf.clip_by_value(swath, 0, 1e10)
        swath = normalize(swath)
        patches = tf.cond(
            tf.reduce_any(tf.is_nan(swath)),
            lambda: tf.expand_dims(tf.zeros(shape), 0),
            lambda: extract_patches(swath),
        )
        return tf.data.Dataset.from_tensor_slices(patches)

    return fn


def load_data(
    data_files,
    fields,
    meta_json,
    shape,
    batch_size,
    normalization,
    read_threads,
    prefetch,
    shuffle_buffer_size,
):
    """Returns the number of channels and a dataset of images.
    See `add_pipeline_cli_arguments` for description of Arguments.
    """
    chans, parser = main_parser(fields, meta_json)
    normalizer = normalizer_fn(normalization)
    shape = (*shape, chans)

    dataset = (
        tf.data.TFRecordDataset(data_files, num_parallel_reads=read_threads)
        .interleave(fused_read_fn(parser, normalizer, shape), cycle_length=read_threads)
        .filter(heterogenous_bands(0.5))  # TODO flag for threshold, also inline
        .apply(shuffle_and_repeat(shuffle_buffer_size))
        .apply(batch_and_drop_remainder(batch_size))
        .prefetch(prefetch)
    )
    return chans, dataset
