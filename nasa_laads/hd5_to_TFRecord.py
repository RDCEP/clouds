'''
Routine to convert MOD06_L2 HDF swath files to TFRecord format
'''

import tensorflow as tf
from pyhdf.SD import SD, SDC



def HDFtoTFRecord(fileURL):
    # Get file and create pyHDF object
    HDFobj = SD(fileURL, SDC.READ)

    datasets_dic = HDFobj.datasets()

    # Retrieve variable space name
    datafields = list()
    shapelist = list()
    # print(datasets_dic['Cloud_Top_Pressure_Infrared'])
    for idx, sds in enumerate(datasets_dic.keys()):
        # print(sds)#idx, sds)
        datafields.append(sds)

        singlefield = HDFobj.select(sds)
        shapelist.append(singlefield[:].shape)

        # print(idx)
        # print(datafields[idx])
        # print(shapelist[idx])

        # print(dimension_dict(datafields[idx],))
        if len(shapelist[idx]) == 1:
            tf_dict = dimension_dict(datafields[idx],max(shapelist[idx]), 1,1,None)
            # print(tf_dict)
        if len(shapelist[idx]) == 2:
            tf_dict = dimension_dict(datafields[idx],max(shapelist[idx]),min(shapelist[idx]),1,None)
            print(tf_dict)
        elif len(shapelist[idx]) == 3:
            tf_dict = dimension_dict(datafields[idx],max(shapelist[idx]),None,min(shapelist[idx]),None)
            # print(tf_dict)
        else:
            # print('Error: HDF dimension with unknown datastructure')
            # print('Tuple length: '+str(len(shapelist[idx])))
            # print(datafields[idx])
            # print(shapelist[idx])
            return

    # For every data element in list return its shapes

    # print(datafields[15])
    # print(shapelist[15])
    # print(len(shapelist[15]))
    # print(shapelist[15][0])
    # print(shapelist[15][1])
    # print(shapelist[15][2])
    # print(min(shapelist[15]))
    # for key in datafields:


    return

def dimension_dict(hdffield, height, width, channels, binaryvalues):
    dictionary = {'hdf_field': hdffield,
                  'features': {
                      'height': height,
                      'width': width,
                      'channels': channels,
                      'binary_raw': binaryvalues
                  }
    }
    return dictionary

# TF Helper functions
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
##################

# Testing routine
url_folder = '/home/rlourenco/'
file_name = 'MOD06_L2.A2017001.0115.061.2017312163804'
file_extension = '.hdf5'
HDFtoTFRecord(url_folder+file_name+file_extension)