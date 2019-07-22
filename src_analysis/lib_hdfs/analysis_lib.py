# _*_ coding : utf-8   _*_

import os
import cv2
import random
import numpy as np
import itertools
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt 
from pylab import *
from matplotlib import patches as mpl_patches
from matplotlib.colors import Colormap
from pyhdf.SD import SD, SDC
from osgeo import gdal

#analytical tools
from sklearn.cluster import AgglomerativeClustering
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#pandas

#multi processing
#import multiprocessing
#from multiprocessing import Pool


def _get_swath(datadir, filename):
    swath = gdal.Open(datadir+'/'+filename)
    data  = swath.ReadAsArray()
    rot_data = np.rollaxis(data, 0, 3)
    print(" Shape ", rot_data.shape)
    return rot_data

def _load_tif_data(mod06_tif_dir, basename, inputfilename):
    mod06_swath =  gdal.Open(mod06_tif_dir+"/"+basename+inputfilename+'.tif')
    print('variable : %s' % inputfilename )
    return  mod06_swath.ReadAsArray()

def cv2_interpolation(array, xsize=0, ysize=0):
    """
    Interpolation by OpenCV for modis06 data
    """
    size = (ysize, xsize)
    resized_img = cv2.resize(array, size)
    return resized_img

def _main_get_itpl_patches(tif_dir, basename, varname, swath, flag_norm=False):
    data = _load_tif_data(tif_dir,basename, varname)
    data = data.astype(np.float64)
    int_data = cv2_interpolation(data, xsize=swath.shape[0], ysize=swath.shape[1])
    patches = _gen_patches(int_data, normalization=flag_norm)
    return patches, int_data

def _get_colors(n=-1, cmap_name='jet'):
    # colormap 
    cmap = cm.get_cmap(cmap_name, n)
    colors = []
    for idx, i in enumerate(range(cmap.N)):
        rgb = cmap(i)[:3]
        #print(idx, matplotlib.colors.rgb2hex(rgb))
        colors += [matplotlib.colors.rgb2hex(rgb)]
    return colors

def get_rand_colors(n=10, cmap_name='jet', _seed = 12356):
    #colormap
    cmap = cm.get_cmap(cmap_name, n)
    colors = []
    for idx, i in enumerate(range(cmap.N)):
        rgb = cmap(i)[:3]
        colors += [matplotlib.colors.rgb2hex(rgb)]
    # shuffle
    random.seed(_seed)
    random.shuffle(colors)
    return colors

def _gen_patches(img, stride=128, size=128, 
                 normalization=True, flag_nan=True, isNoBackground=False):
    """ IF user get patches WITHOUT Normalization
         normalization = False
         Otherwise, vals in patch are normalized
        IF user want to get patches WITHOUT NAN value
         flag_nan=True
         Then, nanvalue will be excluded
    """
    # generate swath again
    swath = img   
    # Fix boolean option now
    if flag_nan:
      swath_mean = np.nanmean(swath, axis=(0,1))
      swath_std = np.nanstd(swath, axis=(0,1))
    else :
      swath_mean = swath.mean(axis=(0,1))
      swath_std = swath.std(axis=(0,1))
    # modify small std value 
    ill_stds = np.where(swath_std < 1.0e-20)[0]
    if len(ill_stds) > 0 :
        print("!====== Ill shape ======!")
        print(np.asarray(ill_stds).shape)
        print(ill_stds)  # coresponding to number of band
        for idx in ill_stds:
          swath_std[idx] += 1.0e-20
    patches = []

    stride = stride
    patch_size = size

    patches = []
    for i in range(0, swath.shape[0], stride):
        row = []
        for j in range(0, swath.shape[1], stride):
            if i + patch_size <= swath.shape[0] and j + patch_size <= swath.shape[1]:
                #p = swath[i:i + patch_size, j:j + patch_size].astype(float)
                if isNoBackground:
                  tmp_p = swath[i:i + patch_size, j:j + patch_size].astype(float)
                  # select only positice patch
                  if not np.all(tmp_p <= 1.0e-5):
                    p = tmp_p
                    if normalization:
                      p -= swath_mean
                      p /= swath_std
                    row.append(p)
                else:
                  p = swath[i:i + patch_size, j:j + patch_size].astype(float)
                  if normalization:
                    p -= swath_mean
                    p /= swath_std
                  row.append(p)
            
                #row.append(p)
        if row:
            patches.append(row)
    # original retuern        
    #return np.stack(patches)
    # Avoid np.stack ValueError if patches = []
    if patches:
      return np.stack(patches)
    else:
      return patches


def _anl_agl(encoder, patches, clusters=0, xsize=128, ysize=128, nband=7):
    # clustering
    #_encs = encoder.predict(patches.reshape((-1, 128, 128, 7)))
    _encs = encoder.predict(patches.reshape((-1, xsize, ysize, nband)))
    method = AgglomerativeClustering(n_clusters=clusters)
    features = _encs.mean(axis=(1,2))
    patches_labels = method.fit_predict(features).reshape(patches.shape[:2])
    return patches_labels

def cluster_plotting2(swath, patches,patches_labels, SHAPE, colors, ncluster=0, target_cluster=0):
    line_width = 2
    fig, a = plt.subplots(figsize=(20,20))
    plt.imshow(swath[:,:,0], cmap="bone")
    rects = []   
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            if patches_labels[i,j] == target_cluster:
                a.add_patch(mpl_patches.Rectangle(
                    (j * SHAPE[0] + line_width , i * SHAPE[1] + line_width ),
                    width=SHAPE[0] - line_width * 2,
                    height=SHAPE[1] - line_width * 2,
                    linewidth=4,
                    edgecolor=colors[patches_labels[i,j]],
                    facecolor="none"
                ))
                centerx = SHAPE[0]*0.5*(2*j+1) #+1
                centery = SHAPE[1]*0.5*(2*i+1) #+1
                # add label as text
                plt.text(centerx, centery, str(patches_labels[i,j]), 
                     fontsize=16,weight='bold', 
                     color=colors[patches_labels[i,j]])
    plt.title(" ## %d cluster ## " %ncluster, fontsize=20)
    plt.show()


def _gen_patch_list(swath, tif_list=[], var_filelist=[],name_list=[], 
                    nfile=-1, file_idx=-1, normalization=False
    ):
    ### gen patch_list
    #TODO; nfile ==> file index or kind of varname
    """ file_idx : index of input list to refer during process; iundex for target filename 
    """
    idx_file = max(nfile, file_idx)  # accept both nfile; file_idx
    for tif_dir, basename, swath in zip( [tif_list[idx_file]], [name_list[idx_file]], [swath]):
        print("==================", tif_dir, "==================")
        patch_list, patch_labels_list = [], []
        for varname in var_filelist:
            _patches, int_data = _main_get_itpl_patches(tif_dir, basename, varname, swath, flag_norm=normalization)
            patch_list += [_patches]
    return patch_list

def _get_laplacian(img, iband=0, imax=8000.00, alpha=255):
    _img = img[:,:,iband]
    img2 = _img/imax*alpha
    return cv2.Laplacian(img2,cv2.CV_64F)


def _get_cosine_sim(xlist):
    """ get cosine similarity for icluster's list 
    """
    array_1d = [ x.flatten() for x in xlist]
    _array_1d = [x.reshape( len(array_1d[0]),1) for x in array_1d]
    cosine_sim = [cosine_similarity(x,y) for x in _array_1d for y in _array_1d if x is not y ]
    return cosine_sim


def _get_cluster_mean(mod09_patches,pointer, patches_list= [], 
                      cluster=-1, normalization=True):
    '''
    mod09_patches : array(x, y, 128, 128, 7) 
    ==> process should mod09_patches.mean(axis=(2,3))[:,:,0] # band 1 (0 in python)
    
    Return 
    ==> Array of Patch-wise Mean data  
    '''
    patch_list = []
    
    def _get_values(array, pointer):
        """
        pointer : 2D array [num, ix;0 and iy;1]
        """
        array_2d = []
        for i in range(len(pointer)):
            ix = pointer[i][0]
            iy = pointer[i][1]
            tmp_array = array[ix,iy]
            array_2d+=[tmp_array]
        return array_2d
    
    # SELECT i clusters' patches
    for ipatches in patches_list:
        patch_list += [ _get_values(ipatches, pointer) ]
    print("     ## check mod09 shape ", mod09_patches.shape )
    mean_patches = _get_values(mod09_patches.mean(axis=(2,3))[:,:,0] , pointer)
    
    
    # normalization and mean
    X_mean_list = []
    sc = StandardScaler()
    print("Shape Patch_list :", np.asarray(patch_list).shape)
    print("Normalization    :", normalization)
    for ipatches in patch_list:
        np_ipatches = np.asarray(ipatches)
        means = np_ipatches.mean(axis=(1,2)).reshape(len(np_ipatches), 1) # compute patch-wise mean
        # normalize patch-wise mean data
        if normalization:
            means_std = sc.fit_transform(means).flatten()
        else:
            means_std = means.flatten()  # no zscore normalization
        X_mean_list += [ means_std ] 
    
    # concatenation of variables
    X_array = np.concatenate([x.reshape(len(x),1) for x in X_mean_list], axis=1)
    print("  + Explanatory Matrix ", X_array.shape)
    print("  + MOD09 Array        ", len(mean_patches) )
    
    return X_array, mean_patches

def cluster_plotting(swath, patches,patches_labels, SHAPE, colors, ncluster=0):
    line_width = 4
    fig, a = plt.subplots(figsize=(20,20))
    plt.imshow(swath[:,:,0], cmap="bone")
    rects = []   
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            a.add_patch(mpl_patches.Rectangle(
                (j * SHAPE[0] + line_width , i * SHAPE[1] + line_width ),
                width=SHAPE[0] - line_width * 2,
                height=SHAPE[1] - line_width * 2,
                linewidth=6,
                edgecolor=colors[patches_labels[i,j]],
                facecolor="none"
            ))
            centerx = SHAPE[0]*0.5*(2*j+1) #+1
            centery = SHAPE[1]*0.5*(2*i+1) #+1
            # add label as text
            plt.text(centerx, centery, str(patches_labels[i,j]), 
                 fontsize=20,weight='bold', 
                 color=colors[patches_labels[i,j]])
    plt.title(" ## %d cluster ## " %ncluster, fontsize=20)
    plt.show()

def _cluster_plotting(swath, patches,patches_labels, SHAPE, colors, ncluster=0):
    line_width = 4
    fig, a = plt.subplots(figsize=(20,20))
    plt.imshow(swath[:,:,0], cmap="bone")
    rects = []   
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            a.add_patch(mpl_patches.Rectangle(
                (j * SHAPE[0] + line_width , i * SHAPE[1] + line_width ),
                width=SHAPE[0] - line_width * 2,
                height=SHAPE[1] - line_width * 2,
                linewidth=1,
                edgecolor=colors[patches_labels[i,j]],
                facecolor="none"
            ))
            centerx = SHAPE[0]*0.5*(2*j+1) #+1
            centery = SHAPE[1]*0.5*(2*i+1) #+1
            # add label as text
            plt.text(centerx, centery, str(patches_labels[i,j]), 
                 fontsize=1,weight='bold', 
                 color=colors[patches_labels[i,j]])
    plt.title(" ## %d cluster ## " %ncluster, fontsize=20)
    plt.show()


def cluster_plotting2(swath, patches,patches_labels, SHAPE, colors, ncluster=0, target_cluster=0):
    line_width = 2
    fig, a = plt.subplots(figsize=(20,20))
    plt.imshow(swath[:,:,0], cmap="bone")
    rects = []   
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            if patches_labels[i,j] == target_cluster:
                a.add_patch(mpl_patches.Rectangle(
                    (j * SHAPE[0] + line_width , i * SHAPE[1] + line_width ),
                    width=SHAPE[0] - line_width * 2,
                    height=SHAPE[1] - line_width * 2,
                    linewidth=4,
                    edgecolor=colors[patches_labels[i,j]],
                    facecolor="none"
                ))
                centerx = SHAPE[0]*0.5*(2*j+1) #+1
                centery = SHAPE[1]*0.5*(2*i+1) #+1
                # add label as text
                plt.text(centerx, centery, str(patches_labels[i,j]), 
                     fontsize=16,weight='bold', 
                     color=colors[patches_labels[i,j]])
    plt.title(" ## %d cluster ## " %ncluster, fontsize=20)
    plt.show()


def mod06_proc_sds(sds_array, variable='sds var'):
    """
    IN: array = hdf_data.select(variable_name)
    """
    array = sds_array.get()
    array = array.astype(np.float64)
    
    # nan process
    nan_idx = np.where( array == sds_array.attributes()['_FillValue'])
    if len(nan_idx) > 0:
        array[nan_idx] = np.nan
    else:
        pass
    
    # radiacne offset
    offset = sds_array.attributes()['add_offset']
    array = array - offset
    
    # radiance scale
    scales = sds_array.attributes()['scale_factor']
    array = array*scales
    
    ### Error Value process
    if variable == 'Cloud_Optical_Thickness':
        err_idx = np.where(array > 100.0) # optical thickness range[0,100] no unit
        array[err_idx] = np.nan
    
    return array
