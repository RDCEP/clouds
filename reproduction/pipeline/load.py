import tensorflow as tf
import json
from tensorflow.contrib.data import shuffle_and_repeat
from tensorflow.contrib.data import parallel_interleave
from tensorflow.contrib.data import batch_and_drop_remainder


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


def load_data(
    data_glob,
    shape,
    batch_size=32,
    read_threads=4,
    shuffle_buffer_size=1000,
    prefetch=1,
    flips=True,
    rotate=False,
    distribute=(1, 0),
    repeat=True
):
    """Returns a dataset of (filenames, coordinates, patches).
    See `add_pipeline_cli_arguments` for argument descriptions.
    """

    def parser(ser):
        features = {
            "shape": tf.FixedLenFeature([3], tf.int64),
            "patch": tf.FixedLenFeature([], tf.string),
            "filename": tf.FixedLenFeature([], tf.string),
            "coordinate": tf.FixedLenFeature([2], tf.int64),
        }
        decoded = tf.parse_single_example(ser, features)
        patch = tf.reshape(
            tf.decode_raw(decoded["patch"], tf.float32), decoded["shape"]
        )
        if rotate:
            angle = tf.random_uniform((), 0, 6.28)
            patch = tf.contrib.image.rotate(patch, angle)
            patch = tf.image.central_crop(patch, 2 ** -0.5)

        patch = tf.random_crop(patch, shape)
        if flips:
            patch = tf.image.random_flip_up_down(tf.image.random_flip_left_right(patch))
        return decoded["filename"], decoded["coordinate"], patch

    dataset = (
        tf.data.Dataset.list_files(data_glob, shuffle=True)
        .shard(*distribute)
        .apply(
            parallel_interleave(
                lambda f: tf.data.TFRecordDataset(f).map(parser),
                cycle_length=read_threads,
                sloppy=True,
            )
        )
    )
    if repeat:
        dataset = dataset.apply(shuffle_and_repeat(shuffle_buffer_size))
    else:
        dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.apply(batch_and_drop_remainder(batch_size)).prefetch(prefetch)
    
    return dataset


def add_pipeline_cli_arguments(p):
    p.add_argument(
        "--data", help="patterns to pick up tf records files", required=True, nargs="+"
    )
    p.add_argument(
        "--shape",
        nargs=3,
        type=int,
        metavar=("h", "w", "c"),
        help="Shape of input image",
        default=(64, 64, 7),
    )

    p.add_argument("--batch_size", type=int, help=" ", default=32)
    p.add_argument("--read_threads", type=int, default=4)
    p.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="Size of prefetch buffers in dataset pipeline",
    )
    p.add_argument("--shuffle_buffer_size", type=int, default=1000)
    p.add_argument("--no_augment_flip", action="store_true", default=False)
    p.add_argument("--no_augment_rotate", action="store_true", default=False)
