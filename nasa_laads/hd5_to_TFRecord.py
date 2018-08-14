import tensorflow as tf

from pyhdf.SD import SD, SDC

file_name = "MOD06_L2.A2017001.0115.061.2017312163804.hd5"
file = SD(file_name, SDC.READ)
