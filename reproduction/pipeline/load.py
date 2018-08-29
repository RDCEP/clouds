import tensorflow as tf
from osgeo import gdal, ogr
import numpy as np
import json


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
