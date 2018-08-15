'''
Routine to convert MOD06_L2 HDF swath files to TFRecord format
'''

import tensorflow as tf
from pyhdf.SD import SD, SDC
from statistics import median
import json
import numpy as np
from mpi4py import MPI
from os import path
from argparse import ArgumentParser
import os
from IPython import embed # DEBUG

def hdf2tfr(hdf_file):
    print(hdf_file)

    data = SD(hdf_file, SDC.READ)
    meta = {}
    tfr_features = {}

    for field in data.datasets().keys():
        values = data.select(field)[:] # Index to get it as np array
        shape = values.shape

        if len(shape) == 1:
            shape = max(shape), 1, 1
        elif len(shape) == 2:
            shape = max(shape), min(shape), 1
        elif len(shape) == 3:
            shape = max(shape), median(shape), min(shape)
        else:
            print("Warining: Encountered rank 4 field %s" % field)
            continue

        meta[field] = shape
        # Wow such boilerplate google... all these types and kwargs are needed?
        tfr_features[field] = tf.train.Feature(
            float_list=tf.train.FloatList(
                value=values.astype(np.float32).flatten()
            )
        )
    example = tf.train.Example(
        features=tf.train.Features(
            feature=tfr_features
        )
    )

    base, _ = path.splitext(hdf_file)
    json_file = base + ".json"
    tfr_file = base + ".tfrecord"

    with open(json_file, "w") as f:
        json.dump(meta, f)

    with tf.python_io.TFRecordWriter(tfr_file) as f:
        f.write(example.SerializeToString())

    print("Recorded %s" % base)



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
            parameters = datafields[idx], max(shapelist[idx]), 1, 1, HDFobj.select(datafields[idx])[:]
            tf_dict = dimension_dict(parameters)
            js_dict = json_dict(parameters)

        if len(shapelist[idx]) == 2:
            parameters = datafields[idx], max(shapelist[idx]), min(shapelist[idx]), 1, HDFobj.select(datafields[idx])[:]
            tf_dict = dimension_dict(parameters)
            js_dict = json_dict(parameters)

        elif len(shapelist[idx]) == 3:
            parameters = datafields[idx], max(shapelist[idx]), median(shapelist[idx]), min(shapelist[idx]), HDFobj.select(datafields[idx])[:]
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
    # print(json_buffer)
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
        hdffield + '_binary_raw': bytes_feature(np.array(binaryvalues).tobytes()) # bytes_feature(np.zeros(1).tobytes()) -- test without data dump
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

# # Testing routine
# url_folder = '/home/rlourenco/'
# file_name = 'MOD06_L2.A2017001.0115.061.2017312163804'
# file_extension = '.hdf5'
# HDFtoTFRecord(url_folder, file_name, file_extension)

p = ArgumentParser()
p.add_argument("--hdf_dir", required=True)
# p.add_argument("--out_dir", required=True)

FLAGS = p.parse_args()
FLAGS.hdf_dir = path.abspath(FLAGS.hdf_dir)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

targets = os.listdir(FLAGS.hdf_dir)
targets = [path.join(FLAGS.hdf_dir, t) for t in targets if t[-4:] == ".hdf"]
targets.sort()
targets = [t for i,t in enumerate(targets) if i % size == rank]
for t in targets:
    # HDFtoTFRecord(FLAGS.out_dir, t, '.hdf')
    hdf2tfr(path.join(FLAGS.hdf_dir, t))
