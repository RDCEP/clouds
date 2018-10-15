import tensorflow as tf
import os
import cv2
import json
import glob
import numpy as np

from osgeo import gdal
from mpi4py import MPI
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import path
from pyhdf.SD import SD, SDC


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def save_example_and_metadata(original, features, meta):
    """Saves tf record from examples and json meta data using original name but
    changing file extension.
    """
    examples = [
        tf.train.Example(features=tf.train.Features(feature=f)) for f in features
    ]

    base, _ = os.path.splitext(original)
    json_file = base + ".json"
    tfr_file = base + ".tfrecord"

    with open(json_file, "w") as f:
        json.dump(meta, f)

    with tf.python_io.TFRecordWriter(tfr_file) as f:
        for example in examples:
            f.write(example.SerializeToString())


def normalized_patches(tif_files, shape, strides, resize):
    """Generates normalized patches.
    """
    for tif_file in tif_files:
        print("Reading", tif_file, flush=True)
        swath = gdal.Open(tif_file).ReadAsArray()
        swath = np.rollaxis(swath, 0, 3)
        if resize is not None:
            swath = cv2.resize(
                swath, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA
            )
        # TODO Flags for other kinds of normalization
        # NOTE: Normalizing the whole (sometimes 8gb) swath will double memory usage by
        # casting it from int16 to float32. Instead normalize and cast patches.
        mean = swath.mean(axis=(0, 1)).astype(np.float32)
        std = swath.std(axis=(0, 1)).astype(np.float32)
        max_x, max_y, _ = swath.shape
        stride_x, stride_y = strides
        shape_x, shape_y = shape

        # Shuffle patches
        coords = []
        for x in range(0, max_x, stride_x):
            for y in range(0, max_y, stride_y):
                if x + shape_x < max_x and y + shape_y < max_y:
                    coords.append((x, y))
        np.random.shuffle(coords)

        for x, y in coords:
            patch = swath[x : x + shape_x, y : y + shape_y]
            # Filter away patches with Nans or if every channel is over 50% 1 value
            # Ie low cloud fraction.
            threshold = shape_x * shape_y * 0.5
            max_uniq = lambda c: max(np.unique(patch[:, :, c], return_counts=True)[1])
            has_clouds = any(max_uniq(c) < threshold for c in range(patch.shape[-1]))
            if has_clouds and not np.isnan(patch).any():
                patch = (patch.astype(np.float32) - mean) / std
                yield tif_file, (x, y), patch


def write_patches(rank, patches, out_dir, patches_per_record):
    """Writes `patches_per_record` patches into a tfrecord file in `out_dir`.
    """
    for i, (filename, coord, patch) in enumerate(patches):
        if i % patches_per_record == 0:
            rec = "{}-{}.tfrecord".format(rank, i // patches_per_record)
            print("Writing to", rec, flush=True)
            f = tf.python_io.TFRecordWriter(os.path.join(out_dir, rec))
        feature = {
            "filename": _bytes_feature(bytes(filename, encoding="utf-8")),
            "coordinate": _int64_feature(coord),
            "shape": _int64_feature(patch.shape),
            "patch": _bytes_feature(patch.ravel().tobytes()),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        f.write(example.SerializeToString())

    print("Rank", rank, "wrote", i + 1, "patches")


def get_args(verbose=False):
    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Turns tif or hdf data into tfrecords. TODO better description.",
    )
    p.add_argument("source_glob", help="Glob of files to convert to tfrecord")
    p.add_argument("out_dir", help="Directory to save results")
    p.add_argument(
        "mode",
        choices=["tif", "hdf", "pptif", "mod021km"],
        help=(
            "`tif`: Turn whole .tif swath into tfrecord. "
            "`hdf`: Turn .hdf swath into tfrecord (respecting fields). "
            "`pptif`: preprocessed_tif, normalize and patchify a tif file."
            "`mod021km` : Imports MOD021KM hdf swath and converts selected bands to HDF"
        ),
    )
    p.add_argument(
        "--fields",
        nargs="+",
        help=(
            "This is only used when translating hdf files, it specifies which fields to "
            "record. If none are provided then all fields are recorded. For MOD021KM files, "
            "using nomenclarture as: bNNG, where NN is the band number (01 - 36) and G stands"
            "for gain (L - Low; H - High). For tif files, all fields are translated as b0..bN"
            "and this flag is ignored."
        ),
    )
    p.add_argument(
        "--shape",
        nargs=2,
        type=int,
        help="patch shape. Only used for pptif",
        default=(128, 128),
    )
    p.add_argument(
        "--resize",
        type=float,
        help="Resize fraction e.g. 0.25 to quarter scale. Only used for pptif",
    )
    p.add_argument(
        "--stride",
        nargs=2,
        type=int,
        help="patch stride. Only used for pptif",
        default=(64, 64),
    )
    p.add_argument(
        "--patches_per_record", type=int, help="Only used for pptif", default=500
    )

    FLAGS = p.parse_args()
    if verbose:
        for f in FLAGS.__dict__:
            print("\t", f, (25 - len(f)) * " ", FLAGS.__dict__[f])
        print("\n")

    FLAGS.out_dir = os.path.abspath(FLAGS.out_dir)
    return FLAGS

def normalized_mod02_patches(hdf_file, out_dir, target_fields):
    """Converts HDF file into a tf record by serializing all fields and	
    names to a record. Also outputs a json file holding the meta data.	
    Arguments:	
        hdf_file: File to convert into a tf record	
        target_fields: List of specified fields to convert	
    """

    file = SD(hdf_file, SDC.READ)

    names = (
        "EV_250_Aggr1km_RefSB",
        "EV_500_Aggr1km_RefSB",
        "EV_1KM_RefSB",
        "EV_1KM_Emissive"
    )

    #TODO
    # extract whole swath
    # rearrange so [height, width, bands (in order)
    #   generalize to other km labels
    # normalize swath
    # extract patches

    fields = [file.select(n)[:] for n in names]

    #Labels for the parsed examples being band number, and eventually gain
    labels_1km = [["b1", "b2"],
              ["b3", "b4", "b5", "b6", "b7"],
              ["b8", "b9", "b10", "b11", "b12", "b13L", "b13H", "b14L", "b14H", "b15", "b16", "b17", "b18", "b19", "b26"],
              ["b20", "b21", "b22", "b23", "b24", "b25", "b27", "b28", "b29", "b30", "b31", "b32", "b33", "b34", "b35",
              "b36"]]

    fields = [
        ("fieldname", [1,2,3]) # bands to extract in order
    ]

    res = np.stack([file.select(f)[bands] for f, bands in fields])
    # channels last
    res = np.rollaxis(res, 0, 3)






    res = np.stack([
        fields[0],
        fields[1],
        fields[2][??],
        fields[2][???],
        fields[3][:5]
                    ], axis = 0)


    yield hdf_file, (x,y), patch

    for n in names:
        print(
            file.select(n)[:].shape)

    # hdf = SD(hdf_file, SDC.READ)
    # meta = {}
    # features = {}
    # for field in hdf.datasets().keys():
    #     if target_fields and field not in target_fields:
    #         continue
    #      data = hdf.select(field)[:]  # Index to get it as np array
    #      # Make sure every field is of rank 4
    #     while len(data.shape) < 3:
    #         data = np.expand_dims(data, -1)
    #     if len(data.shape) > 3:
    #         print(
    #             "Warning, encountered high rank field %s with shape %s"
    #             % (field, data.shape)
    #         )
    #         continue
    #      # Reorder dimensions such that it is longest dimension first
    #     # Assumes height > width > channels
    #     rank_order = list(np.argsort(data.shape))
    #     rank_order.reverse()
    #     data = data.transpose(rank_order)
    #      ty = str(data.dtype)
    #     meta[field] = [data.shape[-1], ty]
    #     features[field + "/shape"] = _int64_feature(data.shape)
    #     features[field + "/type"] = _bytes_feature(ty.encode("utf_8"))
    #     features[field] = _bytes_feature(data.tobytes())
    #  out_file = path.join(out_dir, path.basename(hdf_file))
    #  save_example_and_metadata(out_file, [features], meta)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    FLAGS = get_args(verbose=not rank)
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    targets = []
    for i, f in enumerate(glob.glob(FLAGS.source_glob)):
        if i % size == rank:
            targets.append(os.path.abspath(f))

    if FLAGS.mode == "hdf":
        for t in targets:
            print("Rank", rank, "Converting", t)
            hdf2tfr(t, FLAGS.out_dir, FLAGS.fields)

    elif FLAGS.mode == "tif":
        for t in targets:
            print("Rank", rank, "Converting", t)
            tif2tfr(t, FLAGS.out_dir)

    elif FLAGS.mode == "pptif":
        patches = normalized_patches(targets, FLAGS.shape, FLAGS.stride, FLAGS.resize)
        write_patches(rank, patches, FLAGS.out_dir, FLAGS.patches_per_record)

    elif FLAGS.mode == "mod021km":
        for t in targets:
            print("Rank", rank, "Converting", t)
            #TODO: Link hdf2tfr routine modified to mod021km


    else:
        raise ValueError("Invalid mode")

    print("Rank %d done." % rank, flush=True)
