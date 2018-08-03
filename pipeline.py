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
            f'b{i+1}': tf.FixedLenFeature((side, side), dtype=tf.float32)
            for i in range(n_bands)
        }
        parsed_features = tf.parse_single_example(serialized, features)
        return tf.stack(
            [parsed_features[f'b{i+1}'] for i in range(n_bands)],
            axis=2
        )

    return (tf.data
        .TFRecordDataset(tf.constant(file_names))
        .apply(tf.contrib.data.shuffle_and_repeat(1000))
        .map(parser)
        .prefetch(tf.contrib.data.AUTOTUNE))


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
                for yoff in range(0, rows, side):
                    bands = []
                    for b in range(data.RasterCount):
                        band = data.GetRasterBand(b+1).ReadAsArray(
                            xoff=xoff,
                            yoff=yoff,
                            win_xsize=side,
                            win_ysize=side
                        )
                        bands.append(band)

                    if all(band is not None for band in bands):
                        img = np.stack(bands, axis=-1)
                        if (img != 0).any():
                            yield img
    return gen
