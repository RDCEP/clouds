# _*_ coding:utf-8 _*_
#  
# + Description
#   Max number of `--scale-patch_size` is about ~1.5M patches
#   at Midway cluster computer 100 nodes/6T memory
#
import os
import argparse
import glob
import time
import numpy as np
import random
import sys
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering

###FIXME: add argument parser

#### Settings
p = argparse.ArgumentParser()
p.add_argument(
  '--scale_patch_size',
   help="scale patch size [K] option. 1 = 1,000 patches",
   type=float,
   default=100.0
)
p.add_argument(
  '--n_cluster',
   help="prespecified number of clusters",
   type=int,
   default=10
)
p.add_argument(
  '--inputdir',
   help="input file directory e.g. /home/$USER/data",
   type=str,
   default="/project2/foster/clouds/analysis/output_clouds_feature_2000_2018_validfiles"
)
p.add_argument(
  '--outputdir',
   help="output file directory. Defualt is current directory",
   type=str,
   default=os.getcwd()
)
p.add_argument(
  '--oname',
   help="output file name prefix to distinguish different experiments. e.g. data duration ",
   type=str,
   default='2000-2018_rand'
)
p.add_argument(
  '--modelname',
   help="used DNN model name",
   type=str,
   default='mod02)m01'
)
args = p.parse_args()
### Parameters
print("   Scale  Parameter: %d" % args.scale_patch_size, flush=True)
print("   Input  Directory: %s" % args.inputdir, flush=True)
print("   Output Directory: %s" % args.outputdir, flush=True)
print("   Outfilename     : %s" % args.oname, flush=True)

### Parse Args
scaler=args.scale_patch_size
outputdir=args.outputdir
oname =args.oname
mname =args.modelname
clusters=args.n_cluster

filedir='/project2/foster/clouds/analysis/output_clouds_feature_2000_2018_validfiles' + '/*'
filelist = glob.glob(os.path.abspath(filedir))
#filelist.sort(key=os.path.getmtime) # order of time
#filelist.sort() # order of name

# radomly shuffle input filelist
_seed = random.randint(0,99999)
#_seed = 12356
random.seed(_seed)
random.shuffle(filelist)


print(" Random seed: %d " % _seed, flush=True)

#run 80k patch data through model
### Gen data
array = []
stime = time.time()  # initial time  
npatches = 0
_npatches = 0
used_filelist = []
test_filelist = []
for ifile in filelist:
    tmp_array = np.load(ifile)
    # tmp_array = ['encs_mean', 'clouds_xy']
    encs_mean = tmp_array['encs_mean'] # ndarray[#patches, #dim(128)]
    ni = encs_mean.shape[0] # num of patches
    nk = encs_mean.shape[1] # dimension of DNN model
    nmax = np.amax(encs_mean)
    nmin = np.amin(encs_mean)
    if not np.isnan(nmax) :
      if not np.isnan(nmin):
        print("npatches =", npatches)
        used_filelist.append(ifile)
        if npatches + ni > scaler*1000:
           _npatches = int(scaler*1000 - npatches)
           array += [ encs_mean[:_npatches]]
           break
        elif npatches + ni <= scaler*1000:
           array += [ encs_mean.reshape(ni,nk)]
           npatches += ni
        elif npatches + ni >=  scaler*1000*0.5:
          test_filelist = used_filelist

# open filenamelist to save   
nn = npatches + _npatches
os.makedirs(outputdir, exist_ok=True)
#print(nn[0].shape)
### Save file

# output filename train
ofname="filelist_metadata_train_random-"+str(nn)+'_nc-'+str(clusters)+'_'+str(mname)+'_patches_labels_'+str(oname)
with open(outputdir+"/"+ofname+".txt", 'w') as ofile:
#with open("filelist_metadata_"+str(nn)+".txt", 'w') as ofile:
    ofile.writelines(used_filelist)
ofname="filelist_metadata_test_random-"+str(int(nn/2))+'_nc-'+str(clusters)+'_'+str(mname)+'_patches_labels_'+str(oname)
with open(outputdir+"/"+ofname+".txt", 'w') as ofile:
#with open("filelist_metadata_"+str(nn)+".txt", 'w') as ofile:
    ofile.writelines(test_filelist)

data = np.concatenate(array, axis=0) # Shape of data [#patches, #bottleneck-layer]

##add in ruby's data
ruby_data = np.load('my_features.npy')
both_data = np.concatenate((ruby_data, data))
data = both_data

print("   Data Shape, with ruby's patches [#patches, #Dim.]  :",  data.shape)

### Analysis sklearn Agglomerative clustering
# sklearn - Agglomerative https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
# set method
method = AgglomerativeClustering(n_clusters=clusters)

### Train
# return labels for training data
patches_labels = method.fit_predict(data) ##what i want
print(" NORMAL END : model training ")
print("   # Patches Labels :", patches_labels.shape)

### Save
# Ofile Name --> p-$patch nc-$ncluster_ $model-name $oname
#
np.save(outputdir+'/aggl_p-'+str(int(scaler))+'_nc-'+str(clusters)+'_'+str(mname)+'_train_patches_labels_'+str(oname),
        patches_labels)

etime = (time.time() -stime)  # second
print("   Execution time until train [sec]  : %f" % etime, flush=True)
