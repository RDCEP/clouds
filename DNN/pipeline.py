import tensorflow as tf

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


def tiff_pipeline(file_names, side):
    """Pipeline where the inputs are Tiff files
    """
