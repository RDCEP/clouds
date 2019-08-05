# _*_ coding : utf-8_*_
import os
import sys
import glob
import copy
import numpy as np
import pandas as pd

# square distance
def compute_distance(x,y):
    #c = len(np.squeeze(x))
    return np.sum((x-y)**2)

def compute_centers_dposition(centers_list=[], n_cluster=-1):
    """
    distance i --> i+1
    """
    cdist_array = np.zeros((len(centers_list)-1, n_cluster))
    for i in range(len(centers_list)-1):
        base_centers = np.load(centers_list[i])
        centers = np.load(centers_list[i+1])
        for icluster in range(n_cluster):
                cdist_array[i,icluster] = compute_distance(x=base_centers[icluster], y=centers[icluster])
    return cdist_array

def compute_centers_position(centers_list=[], n_cluster=-1, base_number=0):
    """
    base_number = {0,-1}. if 0; base centroid position is lowest patches. -1 is largest.
    """
    cdist_array = np.zeros((len(centers_list)-1, n_cluster))
    base_centers = np.load(centers_list[base_number])
    fnum = 0
    for idx, ifile in enumerate(centers_list):
        if idx != base_number:
            centers = np.load(ifile)
            for icluster in range(n_cluster):
                cdist_array[fnum,icluster] = compute_distance(x=base_centers[icluster], y=centers[icluster])
            fnum += 1
    return cdist_array

def points2centers(metadata='./x.txt', labeldata='./x.npy', centerdata='x.npy',
                   scaler=1, n_cluster=-1, verbose=0):
    
    if verbose==1:
        print("+ check filename")
        print("meta data  :", os.path.basename(metadata) )
        print("label data :", os.path.basename(labeldata) )
        print("center data:", os.path.basename(centerdata) )
    
    filelist = get_metadata_filelist(metadata)
    patch_data = load_patches(filelist=filelist, scaler=scaler)

    labels = np.load(labeldata)
    ceners = np.load(centerdata)
    clusters = compute_cluster_dists(patch_data, labels, ceners, n_cluster=n_cluster)
    if verbose == 1:
        for i in clusters:
            print("mean dist", np.mean(i))
    return clusters

def compute_cluster_dists(patch_data, labels, centers, n_cluster=-1):
    clusters = []
    dists = []
    for icluster in range(0,n_cluster):
        """
        length of label = idx[0]. idx = [array(....)]
        """
        idx = np.where(labels == icluster)
        ipatches = patch_data[idx]
        dists = []
        for i in range(len(idx[0])):
            dists += [ compute_distance(x=ipatches[i], y=centers[icluster])]
        clusters += [dists]
    return clusters

def load_patches(filelist=[], scaler=10):
    array = []
    npatches = 0
    used_filelist = []
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
                #print("npatches =", npatches)
                used_filelist.append(ifile)
                if npatches + ni > scaler*1000:
                    _npatches = int(scaler*1000 - npatches)
                    array += [ encs_mean[:_npatches]]
                    break
                elif npatches + ni <= scaler*1000:
                    array += [ encs_mean]
                    npatches += ni
    data = np.concatenate(array, axis=0)
    return data

def get_metadata_filelist(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f.readlines():
            iline = line.split(".npz")
            iline.remove('')
            for i in iline: 
                data+=[str(i)]
                
    # get filelist
    npz_filelist = []
    for ifile in data:
        npz_filelist.append(ifile+'.npz')
    return npz_filelist
