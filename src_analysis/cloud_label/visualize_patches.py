#!/usr/bin/env python
# coding: utf-8

# ## Validation analysis against open/closed cells
# ---------------
# This notebook focuses on the validation analysis, which includes inference of trained model with lablled input data and clustering against the output of model.

# --------------
# ### Load module

# In[1]:


import os
import glob
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
from math import ceil,sqrt
from sklearn.manifold import TSNE
import datetime



# In[2]:


## directory where your put lib_hdfs
libdir='/home/rubywerman/clouds/src_analysis/lib_hdfs'


# In[3]:


sys.path.insert(1,os.path.join(sys.path[0],libdir)) # this line helps you to use your own functions in another directory
from alignment_lib import _gen_patches
from alignment_lib import const_clouds_array
from alignment_lib import gen_mod02_img_sigle,  gen_mod35_img_single
from alignment_lib import mod02_proc_sds_single
from alignment_lib import _gen_patches
from alignment_lib import const_clouds_array
from analysis_lib import *
from lib_datesinfo_ruby import *


# ### Load Model

# In[4]:


homedir = '/home/rubywerman/scratch-midway2/lib_hdfs'
datadir = homedir+"/model/m2_02_global_2000_2018_band28_29_31"
step = 100000 # DONOT change so far


# In[5]:


encoder_def = datadir+'/encoder.json'
encoder_weight = datadir+'/encoder-'+str(step)+'.h5'
with open(encoder_def, "r") as f:
    encoder = tf.keras.models.model_from_json(f.read())
encoder.load_weights(encoder_weight)


# ### Load labelled open/closed cell

# Load the date files of your labeled patches here. You can get these files from running and labeling patches in cloud_labeling.ipynb

# In[6]:


#enter the name of the directory containing your dates files, mod02, and mod35 data for your labeled patches
filesdir = "/home/rubywerman/clouds/src_analysis/labeled_data/class_patch_data/"


# The following are patch and cluster objects that make it easy to access and write new data 

# In[7]:


class Patch:        
    def __init__(self, date, isOpen, thirtyFive, zeroTwo, label=None, feature=None, has_coord=False, coords=None):
        self.date = date
        self.isOpen = isOpen
        self.thirtyFive = thirtyFive
        self.zeroTwo = zeroTwo
        self.label = label
        self.feature = feature
        self.has_coord = has_coord
        self.coords = coords
        
    def print_attr(self):
        print("date: " + self.date)
        print("isOpen: " + str(self.isOpen))
        print("label: " + str(self.label))
        if len(self.coords) > 0:
            print("coords: ")
            for i in self.coords:
                print(str(i))   
        
class Cluster:
    def __init__(self, label, patches=None, means=None, std=None, num_open=0):
        self.label = label
        self.patches = patches
        self.means = means
        self.std = std
        self.num_open = num_open


# Run the cell below to load in your list of patch objects from `cloud_labeling.ipynb`

# In[8]:


class_patch_list = np.load(filesdir + '072219.npy')


# In[9]:


#clean faulty patches
class_patch_list = [patch for patch in class_patch_list if type(patch.zeroTwo) is not list]


# In[10]:


print("Number of patches recorded: " + str(len(class_patch_list)))


# In[ ]:


#The following file contains Tak's 80K patches
mypath1 = "/project2/foster/clouds/analysis/output_clouds_feature_2000_2018_validfiles/"


# In[11]:


files = glob.glob(mypath1 + "*.npz")
patches = []
for f in files:
    data = np.load(f)
    lst = data.files
    #data[lst[0]] contains cloud patches np.array(#of patches, 128) 
    patches.append(data[lst[0]])


# In[12]:


#put all cloud patch np.arrays into one np.array
all_patches = patches[0]
for p in patches[1:]:
    all_patches = np.concatenate((all_patches, p), axis=0)


# In[13]:


all_patches.shape


# ### Run Analysis
# 
# we will use a type of hierarchical clusering called `Agglometative clustering` 

# How to donwload agglomerative [sklearn aggl](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)  
# 
# `from sklearn.cluster import AgglomerativeClustering`

# ###### Inference

# In[14]:


encs_list = []
for patch in class_patch_list:
    i = patch.zeroTwo
    if type(i) is not list:
        ix, iy = i.shape[:2]
        encs = encoder.predict(i.reshape(ix * iy, 128,128,6))
        encs_list += [encs.mean(axis=(1,2))]


# In[15]:


features = np.concatenate(encs_list, axis=0)


# In[16]:


print(features.shape)  # make sure, the shape is [#number of patches, 128]


# In[17]:


both_features = np.concatenate((features, all_patches))


# In[18]:


both_features.shape


# ##### Clustering

# In[19]:


from sklearn.cluster import AgglomerativeClustering


# In[21]:


# N in [2, inf), you can change this number but save the result differently
num_clusters = 80


# In[22]:


clustering = AgglomerativeClustering(num_clusters)


# In[23]:


#turns any NAN values to 0 so code doesn't crash
cleaned_features = np.nan_to_num(both_features)


# In[26]:


#generate clustering data
label = clustering.fit_predict(cleaned_features)
np.save('/home/rubywerman/clouds/src_analysis/cloud_label/label_data.npy', label)

