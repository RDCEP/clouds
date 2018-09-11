import tensorflow as tf
import json
from tensorflow.contrib.data import shuffle_and_repeat
from tensorflow.contrib.data import parallel_interleave


def tfr_parser_fn(fields, meta_json, saved_as_bytes=True):
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
        return tf.concat(res, axis=2), sh

    return chans, parser


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


def normalizer_fn(normalization):
    """Returns a function that performs `normalization` to an image tensor.

    FIXME: This does not mask pixels with no clouds in the calculation of
    moments, biasing mean and variance, We want to normalize by conditional
    mean and conditional variance (conditional on 'pixel with a cloud').
    """
    normalization = normalization.lower()

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


def heterogenous_bands(img, threshold=0.5):
    """Returns False if a band in the image has too much of a single value is `threshold`
    fraction of the image. Presumably this value represents no cloud as clouds will be
    more heterogenous.
    """
    has_data = []
    for band in tf.unstack(img, axis=-1):
        _, _, count = tf.unique_with_counts(tf.reshape(band, [-1]))
        has_data.append(tf.reduce_max(count) / tf.size(band) < threshold)
    return tf.logical_and(tf.reduce_all(tf.is_finite(img)), tf.reduce_any(has_data))


def patch_reader_fn(parse, normalize, shape):
    """Returns a function that parses a swath and extracts normalized and labeled patches.
    Args:
        parse: Function that takes serialized TFRecord and outputs a 3d tensor swath
        normalize: Function that normalizes each channel of the swath
        shape: Desired shape of each patch in the swath
    Returns:
        patch_reader: Function that parses and normalizes a serialized tfrecord by
        applying `parse` and `normalize`, and outputs patches and their coordinates from /
        in the swath.

    This is actually a lot faster than non-fused version. Perhaps because TF uses threads
    for each map / flatmap / apply and there is a lot of copying or context switching
    otherwise.
    """
    height, width, chans = shape

    def patch_reader(ser):
        swath, sh = parse(ser)
        swath_x, swath_y = sh[0], sh[1]
        # Half overlapping patches TODO flag to control this?
        stride_x, stride_y = width // 2, height // 2
        rows, cols = swath_x // stride_x - 1, swath_y // stride_y - 1

        swath = tf.clip_by_value(swath, 0, 1e10)
        swath = normalize(swath)
        patches = tf.extract_image_patches(
            images=tf.expand_dims(swath, 0),
            ksizes=[1, height, width, 1],
            strides=[1, stride_x, stride_y, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [-1, height, width, chans])

        def label(num, data):
            row = num // cols
            col = num % cols
            return (row * stride_x, col * stride_y), data

        return (
            tf.data.Dataset.from_tensor_slices(patches)
            .apply(tf.contrib.data.enumerate_dataset())
            .filter(lambda num, patch: heterogenous_bands(patch))
            .map(label)
        )

    return patch_reader


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
    """Returns the number of channels and a dataset of (sourcefile, coordinate, patch).
    See `add_pipeline_cli_arguments` for description of Arguments.
    """
    chans, parser = tfr_parser_fn(fields, meta_json)
    normalizer = normalizer_fn(normalization)
    shape = (*shape, chans)
    dataset = (
        tf.data.Dataset.from_tensor_slices(data_files)
        .apply(shuffle_and_repeat(10000))
        .apply(
            parallel_interleave(
                lambda file_name: (
                    tf.data.TFRecordDataset(file_name)
                    .flat_map(patch_reader_fn(parser, normalizer, shape))
                    .map(lambda coord, patch: (file_name, coord, patch))
                ),
                cycle_length=read_threads,
                sloppy=True,
            )
        )
        # Shuffle again because each swath yields 1000s of very correlated patches
        .shuffle(shuffle_buffer_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(prefetch)
    )
    return chans, dataset
