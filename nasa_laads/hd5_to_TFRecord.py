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

        print(datafields[idx])
        print(shapelist[idx])

    # For every data element in list return its shapes

    # for key in datafields:


    return



# Testing routine
file_name = '/home/rlourenco/MOD06_L2.A2017001.0115.061.2017312163804.hdf5'
HDFtoTFRecord(file_name)