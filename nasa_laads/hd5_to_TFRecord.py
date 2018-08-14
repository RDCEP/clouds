'''
Routine to convert MOD06_L2 HDF swath files to TFRecord format
'''

import tensorflow as tf
from pyhdf.SD import SD, SDC
from statistics import median
import json
import numpy as np


def HDFtoTFRecord(url_folder, file_name, file_extension):
    fileURL = url_folder + file_name + file_extension
    # Get file and create pyHDF object
    HDFobj = SD(fileURL, SDC.READ)

    datasets_dic = HDFobj.datasets()

    # Retrieve variable space name
    datafields = list()
    shapelist = list()

    tfrecords_filename = file_name + '.TFRecord'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    json_buffer = {file_name: {}}
    record_dict = {}
    for idx, sds in enumerate(datasets_dic.keys()):

        datafields.append(sds)

        singlefield = HDFobj.select(sds)
        shapelist.append(singlefield[:].shape)

        if len(shapelist[idx]) == 1:
            parameters = datafields[idx], max(shapelist[idx]), 1, 1, None
            tf_dict = dimension_dict(parameters)
            js_dict = json_dict(parameters)

        if len(shapelist[idx]) == 2:
            parameters = datafields[idx], max(shapelist[idx]), min(shapelist[idx]), 1, None
            tf_dict = dimension_dict(parameters)
            js_dict = json_dict(parameters)

        elif len(shapelist[idx]) == 3:
            parameters = datafields[idx], max(shapelist[idx]), median(shapelist[idx]), min(shapelist[idx]), None
            tf_dict = dimension_dict(parameters)
            js_dict = json_dict(parameters)

        # Transient memory
        transient_record_dict = record_dict
        transient_record_dict.update(tf_dict)
        record_dict = transient_record_dict

        transient_js_dict = json_buffer[file_name]
        transient_js_dict.update(js_dict)
        json_buffer[file_name] = transient_js_dict
        ###################

    # print(record_dict)
    print(json_buffer)
    features_TF_obj = tf.train.Features(feature=record_dict)
    example = tf.train.Example(features=features_TF_obj)
    writer.write(example.SerializeToString())

    # dump json object
    with open(file_name+'.json', 'w') as outfile:
        json.dump(json_buffer, outfile, ensure_ascii=False)

    return


def dimension_dict(tuple):
    hdffield=tuple[0]
    binaryvalues=tuple[4]
    dictionary = {
        hdffield + '_binary_raw': bytes_feature(np.zeros(1).tobytes())
    }

    return dictionary


def json_dict(tuple):
    hdffield = tuple[0]
    height = tuple[1]
    width = tuple[2]
    channels = tuple[3]
    dictionary = {
        hdffield + '_height': height,
        hdffield + '_width': width,
        hdffield + '_channels': channels
    }
    return dictionary


# TF Helper functions
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


##################

# Testing routine
url_folder = '/home/rlourenco/'
file_name = 'MOD06_L2.A2017001.0115.061.2017312163804'
file_extension = '.hdf5'
HDFtoTFRecord(url_folder, file_name, file_extension)
