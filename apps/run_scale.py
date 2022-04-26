#
# Code to run DAG workflow using Parsl
# original source : build-inference-framework.ipynb
import os
import sys
import glob
import time
import numpy as np
from lib4modis import loader
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import parsl
from parsl.app.app   import python_app, bash_app
from parsl.config    import Config
from parsl.channels  import LocalChannel
from parsl.providers import LocalProvider, SlurmProvider, CobaltProvider
from parsl.launchers import AprunLauncher, SrunLauncher
from parsl.addresses import address_by_hostname
from parsl.executors import HighThroughputExecutor 
from parsl.monitoring.monitoring import MonitoringHub

"""
Advice to choose optimal number of worker for Parsl #1
  cores_per_worker=1 leads to 28 workers, which means that there would be 28 func(input=XXX) running concurrently.
  if single threaded, then cores_per_worker=1 should be fine
  if you want to be safe, set cores_per_worker=2 , then there would be 14 workers
  # scale up guidline
  142 files is roughly 6 nodes * 28 workers
  start from 1 node to 6 nodes
  good sanity check is 1 node - 10 files

Advice #2
  you can always change the config to modify the number of cores for each worker 
  (each worker runs one function at a time)
  
  For example, if your function requests 1 core, then you can set 
    - cores_per_worker=1 or 
    - max_workers=NUMBER OF CORES OF the machine
  if you function request n cores (multi-thread inside your function), then you can 
    - cores_per_worker=n  or 
    - max_workers=# of cores / n)
  Note that when set cores_per_worker = 2, parsl does not gurantee that a function runs on 2 cores.
  Assume running case on theta (64cores KNL), then parsl would put at most 64/2 = 32 functions on each node. 
  However, each of these 32 functions is not necessary to use 2 cores.
  Number of used cores depends on how you set your functions.
  Some functions may use 4 cores and some may use only 1 core.

Theta specific config
  https://github.com/h5py/h5py/issues/1101
Monitering config
  Need to install following two libraries;
  - pip install SQLAlchemy
  - pip install sqlalchemy-utils
"""


@python_app
def patch_creation(filepath=None, filenames=None,instrument='terra',
                   ems_band=["20","28","29",'31'],
                   stride=128, patch_size=128, channels=6,
                   mod35_datadir='./',prefix=None, 
                   mod03_datadir='./',prefix03=None,
                   mod06_datadir='./',prefix06=None,
                   outputdir='.',
                   thres_ocean_frac=0.99,thres_cloud_frac=0.3,
                   ocean_only=False,
                   global_mean_file=None, global_stdv_file=None,
                   centroids_filename=None,
                   height=128, width=128,
                   model_datadir="/home/tkurihana/rotate_invariant/stepbystep/transform/output_model", 
                   expname="67011582",
                   layer_name="leaky_re_lu_23",
                   nclusters=12,
                   param_keys=["Cloud_Optical_Thickness","Cloud_Phase_Infrared_1km","cloud_top_pressure_1km","Cloud_Effective_Radius"],
                   nlat=181, nlon=360, 
                   isAppend=False,
        ):
    import os
    import re
    import cv2
    import math
    import time
    import json
    import glob
    import pickle
    import numpy as np
    import scipy as sc
    import pandas  as pd
    import netCDF4 as nc
    from scipy.stats import mode
    from pyhdf.SD import SD, SDC
    from lib4modis import loader # myown analysis module loaded on tf2-gpu
    from collections import OrderedDict
    from collections import defaultdict
    import tensorflow as tf
    from tensorflow.python.keras.models import Model
    from sklearn.metrics.pairwise import euclidean_distances
    from numba import jit
    
    def get_masks(rpatch_size, channels):
        mask = np.zeros((rpatch_size, rpatch_size), dtype=np.float64)
        cv2.circle(mask, center=(rpatch_size // 2, rpatch_size // 2),
                radius=rpatch_size//2, color=1, thickness=-1)
        mask = np.expand_dims(mask, axis=-1)
        #  multiple dimension
        mask_list = [ mask for i in range(channels)]
        masks = np.concatenate(mask_list, axis=-1)
        return masks
    
    def gen_sds(ifile, ref_var='EV_500_Aggr1km_RefSB', ems_var='EV_1KM_Emissive',
            ref_bands=[], ems_bands=[] ):

        ref_sds = loader.read_hdf(ifile, varname=ref_var)
        ems_sds = loader.read_hdf(ifile, varname=ems_var)
        # Abort stopper for HDF-open error    
        if ref_sds is None:
            raise ValueError("ref_sds is None: Check file")  
        #FIXME Probably this second if statement is not necessary.
        elif ems_sds is None:
            raise ValueError("ems_sds is None: Check file")  

        ref_array, cref_bands = loader.proc_sds(ref_sds)
        ems_array, cems_bands = loader.proc_sds(ems_sds)
        # data concatenation
        swath = loader.aug_array(ref_array, ems_array, 
                  ref_bands=ref_bands, ems_bands=ems_bands,
                  cref_bands=cref_bands, cems_bands=cems_bands)
        return ifile, swath
    
    def gen_patches(swaths, stride=128, patch_size=128, channels=6,
                 gmean=None, gstdv=None,nsigma=2,
                 mod35_datadir='./', mod06_datadir='./',thres=0.3, 
                 ocean_thres=0.999,ocean_only=False, 
                 prefix='MOD35_L2.A'):
            """
            """
            fname, swath = swaths
            max_x, max_y, _ = swath.shape
            # gen coordination pointers
            coords = []
            for x in range(0, max_x, stride):
                for y in range(0, max_y, stride):
                    if x + patch_size < max_x and y + patch_size < max_y:
                        coords.append((x, y))
            # Get MOD06 data instead of MOD35
            m06_file = loader.get_filepath(fname, mod06_datadir, prefix=prefix)
            try:
              hdf_m06 = loader.SD(m06_file, SDC.READ)
            except Exception as e:
              hdf_m06 = None
            if hdf_m06 is None:
                print(f"no value in MOD06 files: Check file directory {fname} ", flush=True)
                yield fname, None, None

            clouds_mask_img = loader.gen_mod06_cloud_img(hdf_m06,byte_index=0)

            # generate ocean mask
            ocean_mask_img = loader.gen_mod06_ocean_img(hdf_m06,byte_index=0) 

            for i, j in coords:
                patch = swath[i:i + patch_size, j:j + patch_size]
                if not np.isnan(patch).any():
                    clouds_patch, clouds_flag = loader.translate_const_clouds_array(
                      patch, clouds_mask_img, thres=thres, coord=(i,j)
                    )
                    if clouds_flag:
                        if ocean_only:
                            ocean_clouds_patch, ocean_flag = loader.translate_const_ocean_array(
                              clouds_patch, ocean_mask_img, thres=ocean_thres, coord=(i,j)
                            )
                            # apply cloud mask and scale operator 
                            if ocean_flag:
                                _,_,mocean_clouds_patch = loader.make_scale_mask_operator(
                                    ocean_clouds_patch,fname,coordinate=(i,j), 
                                    gmean=gmean, gstdv=gstdv, channels=channels,nsigma=nsigma
                                )
                                yield fname, (i, j), mocean_clouds_patch
                        else:
                            _,_,mclouds_patch = loader.make_scale_mask_operator(
                                clouds_patch,fname,coordinate=(i,j), 
                                gmean=gmean, gstdv=gstdv, channels=channels,nsigma=nsigma
                            )
                            yield fname, (i, j), mclouds_patch
    
        
    def gen_latlon(fname=None,mod03_datadir=None, prefix=None):
        """fname : mod02 name
        """
        def proc_coord(sds_array):
            """ sds_array = hdf_m03.select("Longitude")
            """
            array = sds_array.get()
            array = array.astype(np.float32)
            nan_idx = np.where( array == sds_array.attributes()['_FillValue'])
            if len(nan_idx) > 0:
              array[nan_idx] = np.nan
            return array

        # Get MOD03 data
        m03_file = loader.get_filepath(fname, mod03_datadir, prefix=prefix)
        if m03_file is None:
          return None, None
        try:
          hdf_m03 = SD(m03_file, SDC.READ)
        except Exception as e:
          hdf_m03 = None
        if hdf_m03 is None:
            print("no value in MOD03 files: Check file directory", flush=True)
            return None, None
        lats = proc_coord(hdf_m03.select("Latitude") )
        lons = proc_coord(hdf_m03.select("Longitude") )
        return lats, lons

    def mod06_proc_sds(hdf_data, variable='sds var'):
        """ migrate from clouds/src_analysis/lib/analysis_lib
        IN: array = hdf_data.select(variable_name)
        """
        sds_array = hdf_data.select(variable)
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
        
        # Error Value process
        if variable == 'Cloud_Optical_Thickness':
            err_idx = np.where(array > 100.0) # optical thickness range[0,100] no unit
            array[err_idx] = np.nan
        
        return array

    def gen_physics(fname,mod06_datadir, prefix=None,param_keys=[]):
        # Get MOD06 data
        m06_file = loader.get_filepath(fname, mod06_datadir, prefix=prefix)
        if m06_file is None:
          return {}
        try:
          hdf_m06 = SD(m06_file, SDC.READ)
        except Exception as e:
          hdf_m06 = None
        if hdf_m06 is None:
            print(f"no value in MOD06 files: Check file directory", flush=True)
            return {} 
    
        # slicing
        param_dict = {}
        for key in param_keys:
          param_dict[key] = mod06_proc_sds(hdf_m06, variable=key)
        return param_dict


    def compute_stats(x,key=None):
        """ compute 1st, 2nd moment and skew and Fisher (excess) kurtosis """
        stats_dict = {}
        if key != "Cloud_Phase_Infrared_1km":
            stats_dict['mean'] = np.nanmean(x)
            stats_dict['std'] = np.nanstd(x)
        elif key == "Cloud_Phase_Infrared_1km":
            stats_dict['mode'] = mode(x.ravel())[0][0]
            stats_dict['clear-sky'] = len(np.argwhere(x.ravel()==0))
            stats_dict['liquid'] = len(np.argwhere(x.ravel()==1))
            stats_dict['ice'] = len(np.argwhere(x.ravel()==2))
            stats_dict['undef'] = len(np.argwhere(x.ravel() >= 3))
        else:
          pass
        return stats_dict

    def save_as_pickle(obj, outputdir,filename):
        if os.path.exists(os.path.join(outputdir, filename)):
          os.remove(os.path.join(outputdir, filename))
        with open(os.path.join(outputdir, filename), 'wb') as f:
          pickle.dump(obj, f)

    def save_as_json(obj, outputdir,filename):
        if os.path.exists(os.path.join(outputdir, filename)):
          os.remove(os.path.join(outputdir, filename))
        with open(os.path.join(outputdir, filename), 'w') as f:
          json.dump(obj, f)

    ###################################################################
    ## START JOB
    ###################################################################

    # select ref band of either terra or aqua due to striping issue
    refband = "6" if instrument.lower() == 'terra' else "5" 
    fname = os.path.join(filepath, filenames)


    # read MOD02 : change from generator to return objenct
    swaths  = gen_sds(fname, 
                      ref_var='EV_500_Aggr1km_RefSB', ems_var='EV_1KM_Emissive',
                      ref_bands=[refband,"7"], ems_bands=ems_band)
    if swaths is None:
        print(f"FILE DOES NOT EXIST : {fname}", flush=True)
        return [0]
    # read MOD03
    lats, lons = gen_latlon(fname,mod03_datadir, prefix=prefix03) 
    
    # read MOD06
    param_dict = gen_physics(fname,mod06_datadir, prefix=prefix06,param_keys=param_keys) 
     
    # read gmean and gstdv
    global_mean = np.load(global_mean_file)    
    global_stdv = np.load(global_stdv_file)    

    niter = 0
    patches_list = []
    coords = []
    glats= []
    glons= []
    stats= []
    timers = []
    
    patches = None
    timers.append(time.time()) # start timer here
    if lats is not None:
        # declare generator here
        patch_info = gen_patches(swaths, 
                              stride=stride, patch_size=patch_size, channels=6,
                              gmean=global_mean,
                              gstdv=global_stdv,
                              mod06_datadir=mod06_datadir, 
                              thres=thres_cloud_frac,prefix=prefix06,
                              ocean_thres=thres_ocean_frac,
                              ocean_only=ocean_only,
        )

        while True:
            try:
                # patch creation
                _fname, coord, patch = next(patch_info)
                #print(np.min(patch,axis=(0,1)), np.max(patch,axis=(0,1))) #;exit(0) # for debug
                if niter == 0:
                    if patch is None:
                      break
                    patches = np.expand_dims(patch, axis=0)
                else:
                    patches = np.concatenate([patches,np.expand_dims(patch, axis=0)], axis=0)
                coords.append(coord)
                
                # metadata association from MOD03
                glats.append(lats[coord[0]:coord[0]+patch_size,coord[1]:coord[1]+patch_size])
                glons.append(lons[coord[0]:coord[0]+patch_size,coord[1]:coord[1]+patch_size])

                # patch-based stats values until 4th moment
                # use statsmodels.sandbox.distributions.extras.pdf_mvsk to retreive original distribution
                stats_dicts = {}
                for key in param_keys:
                  m06_patch = param_dict[key][coord[0]:coord[0]+patch_size,coord[1]:coord[1]+patch_size]
                  stats_dicts[key] = compute_stats(m06_patch,key)
                stats.append(stats_dicts)

                niter+=1
            except StopIteration:
                print('### NORMAL END : Patch creation ###\n')
                break

    # time end patch and metadata creation 
    timers.append(time.time())
    # return if file does not contain any information
    if patches is None:
        return timers
    ############################################################

    def load_latest_model(model_dir, mtype):
        latest = 0, None
        # get trained wegiht 
        for m in os.listdir(model_dir):
            if ".h5" in m and mtype in m:
                epoch = int(m.split("-")[1].replace(".h5", ""))
                latest = max(latest, (epoch, m))

        epoch, model_file = latest

        if not os.listdir(model_dir):
            raise NameError("no directory. check model path again")

        print(" Load {} at {} epoch".format(mtype, epoch))
        model_def = model_dir+'/'+mtype+'.json'
        model_weight = model_dir+'/'+mtype+'-'+str(epoch)+'.h5'
        with open(model_def, "r") as f:
            model = tf.keras.models.model_from_json(f.read())
        model.load_weights(model_weight)

        return model
    
    # import trained model
    model_dir = os.path.join(model_datadir,str(expname) )
    encoder = load_latest_model(model_dir, mtype='encoder')
    decoder = load_latest_model(model_dir, mtype='decoder')
    
    # run inference
    pred = None
    if patches is not None:
        imgs_tf = tf.image.resize(patches,(height,width))
        pred = encs =  encoder.predict(imgs_tf)
        if layer_name:
            model = decoder
            rep  = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            pred = rep.predict(encs)
        n,h,w,c = pred.shape
        pred = pred.reshape(n,h*w*c) # reshape for distance input 
    # time end inference
    timers.append(time.time())

    # predict label assignment
    centroids=np.load(centroids_filename)
    def distance(x,_centroids):
        """ x, y : (n_samples_X, n_features)
        """
        delta = euclidean_distances(x.reshape(1,-1), _centroids)
        label = np.argmin(delta, axis=1)[0]
        return label

    labels = None
    if pred is not None:
      labels = list(map(lambda x : distance(x,centroids), pred) )
    # time end prediction
    timers.append(time.time())

    # 1-deg aggregation
    @jit
    def grid_estimator(x):
        """
             diff1        diff2
            (floor) (      ceil          )   
            | -----x----- | ------------ |
            32            32.5           33
         or
           -33           -32.5          -32
        """
        diff1 = abs( x - math.floor(x) ) 
        diff2 = abs( x - math.ceil(x) ) 
        ex = math.ceil(x) if diff1 > diff2 else math.floor(x)
        return int(ex)

    def aggregate_operator(labels, glats, glons, reso=1.0):
        aggr_dict = defaultdict(list)
        #print(type(labels),type(glats), type(glons), flush=True); exit(0)
        for label, _lons, _lats in zip(labels, glons, glats):
            for ilon, ilat in zip(np.array(_lons).ravel(),np.array(_lats).ravel()) :
                if not np.isnan(ilon) and not np.isnan(ilat):
                    elon = grid_estimator(ilon)
                    elat = grid_estimator(ilat)
                    aggr_dict[f"{elon},{elat}"].append(label)
        return aggr_dict

    def aggregate_param_operator(param, lats, lons, patch_size=128):
        aggr_dict = defaultdict(list)
        for ival, ilon, ilat in zip(param.ravel(), lons.ravel(), lats.ravel()):
            elon = grid_estimator(ilon)
            elat = grid_estimator(ilat)
            aggr_dict[f"{elon},{elat}"].append(ival)
        return aggr_dict

    def most_frequent(List):
        return max(set(List), key = List.count)

    def aggregate_label_counter(mydict):
        label_dict = dict()
        for key, List in mydict.items():
          label_dict[key] = most_frequent(List)
        return label_dict
    
    def aggregate_param_counter(mydict,param_key):
        label_dict = dict()
        if param_key != "Cloud_Phase_Infrared_1km":
            for key, List in mydict.items():
              label_dict[key] = {'mean': np.nanmean(List), 'std': np.nanstd(List) }
        else:
            for key, List in mydict.items():
              #label_dict[key] = {'mean': np.nanmean(List), 'std': np.nanstd(List) }
              x = np.asarray(List)
              label_dict[key] =  {
                  'mode': mode(List)[0][0] ,
                  'clear-sky':len(np.argwhere(x.ravel()==0)),
                  'liquid':len(np.argwhere(x.ravel()==1)),
                  'ice':len(np.argwhere(x.ravel()==2)),
                  'undef':len(np.argwhere(x.ravel() >= 3)),
              }
        return label_dict

    def _run_groupby(df, lonbins=None, latbins=None, param_key=None):
        groups = df.groupby(pd.cut(df.Longitude, lonbins))
        aggr_dict = {}

        if param_key != "Cloud_Phase_Infrared_1km":
            for key, group in groups:
                aggr_dict[key] = group.groupby(pd.cut(group.Latitude, latbins)).myvar.mean()
        else:
            for key, group in groups:
                aggr_dict[key] = group.groupby(pd.cut(group.Latitude, latbins)).myvar.agg(
                                    lambda x:x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan )
        return aggr_dict


    def param_groupby_operator(array, param_key, lats, lons):
        """ group values for specific resolution
            MOD06 param has pixel-wise data capable of using pandas
            IN:
              array ; np.2d array (2030, 1354)
            Reference:
              https://stackoverflow.com/questions/39254704/pandas-group-bins-of-data-per-longitude-latitude
        """
        # bind data into one dataframe
        _df = pd.DataFrame(
                np.concatenate(
                    [np.expand_dims(array.ravel(),axis=1),
                     np.expand_dims(lons.ravel(),axis=1),
                     np.expand_dims(lats.ravel(),axis=1),
                    ], axis=1
                )
            )
        _df.columns = ['myvar', 'Longitude', 'Latitude']
        # drop nan table
        df = _df.dropna(how='any')
        if df.empty:
          return None

        # group by setup
        lonmin = math.floor(df.Longitude.min(skipna=True)) - .5  # E41.5 - E42.5 --> 42
        lonmax = math.ceil(df.Longitude.max(skipna=True))  + .5 
        lonbins_1deg  = np.linspace(lonmin, lonmax, int(abs(lonmax - lonmin))+1 )

        latmin = math.floor(df.Latitude.min(skipna=True))  - .5
        latmax = math.ceil(df.Latitude.max(skipna=True))   + .5
        latbins_1deg  = np.linspace(latmin, latmax, int(abs(latmax - latmin))+1 )
        
        # groupby run
        aggr_dict = _run_groupby(df, lonbins=lonbins_1deg, latbins=latbins_1deg,param_key=param_key)
        return aggr_dict

    def label_retreive_fn(aggr_labels):
        alons=[]
        alats=[]
        alabels=[]
        for key,_label in aggr_labels.items():
            lonlat=key.split(",")
            alons.append(int(lonlat[0]) )
            alats.append(int(lonlat[1]) )
            alabels.append(int(_label) )
        aggr_dict = {"Longitude":alons, "Latitude":alats, "Label":alabels }
        return aggr_dict

    def value_latlon_mapper(mydict):
        lons = []
        for idx, (ilon,val) in enumerate(mydict.items()) :
            lons.append(ilon.left)
            if idx == 0:
                # make latitude
                lats = []
                for i in val.index:
                    lats.append(i.left)
                values = np.expand_dims(np.asarray(val),axis=0)
            else:
                values = np.concatenate(
                    [ values, np.expand_dims(np.asarray(val),axis=0)],axis=0 )

        # re-coordinate to map in matplotlib
        cmatrix = values[:,::-1].T  
        clons, clats = np.meshgrid(np.array(lons)+.5, np.array(lats[::-1])+.5 ) # assume 1deg
        map_dict = {'param':cmatrix, 'Longitude':clons, 'Latitude':clats}
        return map_dict

    @jit
    def latlon_helper(ilat, ilon):
        if ilon < 0:
            ilon += 360
        elif ilon >=360:
            ilon -= 360
        ilat += 90
        if ilat >= 181:
            ilat -= 180
        elif ilat < 0:
            ilat += 180
        return ilat, ilon

    def _label_aggr_save_helper(mydict, nlat=181,nlon=360):
        label_aggr_array = np.zeros((nlat,nlon)).astype(np.float32)
        label_aggr_array[:,:] = np.nan

        alons = mydict['Longitude']
        alats = mydict['Latitude']
        alabels = mydict['Label']
        for ilon, ilat, ilabel in zip(alons, alats, alabels):
            # outlier
            _ilat, _ilon = latlon_helper(ilat, ilon)
            #assign
            label_aggr_array[int(_ilat), int(_ilon)] = ilabel
        return label_aggr_array


    @jit
    def assign_param_aggr_array(param_aggr_array, lons, lats, values):
        for _lons, _lats, _values in zip( lons, lats, values ):
           for ilon, ilat, ival in zip(_lons, _lats, _values):
               # outlier
               _ilat, _ilon = latlon_helper(ilat, ilon)
               #assign
               param_aggr_array[int(_ilat), int(_ilon)] = ival
        return param_aggr_array


    def _param_aggr_save_helper(mydict, nlat=181,nlon=360):
        """ https://numba.readthedocs.io/en/stable/reference/pysupported.html?highlight=dictionary#typed-dict
        """
        map_array_dict = dict()
        for key, nest_dict in mydict.items():
            param_aggr_array = np.zeros((nlat,nlon)).astype(np.float32)
            param_aggr_array[:,:] = np.nan
            try:
                param_aggr_array = assign_param_aggr_array(param_aggr_array, 
                          nest_dict['Longitude'], nest_dict['Latitude'],nest_dict['param']
                )
            except Exception as e:
              pass

            map_array_dict[key] = param_aggr_array
        return map_array_dict


    def add_save_as_netcdf(obj, outputdir,filename, patch_size=128, channels=6, nlon=360, nlat=181,nclusters=12):
        # Add write      

        rootgrp = nc.Dataset(os.path.join(outputdir, filename), 'a', format='NETCDF4')
        # *aggregation data*
        # - label_aggregation
        label_aggr_array_ = rootgrp.createVariable(f'nc-{nclusters}-label_1deg_aggregation', 'f4', 
                                ('Latitude', 'Longitude',))
        label_aggr = _label_aggr_save_helper(obj['label_aggregation'], nlat=nlat,nlon=nlon)
        label_aggr_array_[:,:] = label_aggr.astype(np.float32)

        # close file
        rootgrp.close()    



    def save_as_netcdf(obj, outputdir,filename, patch_size=128, channels=6, nlon=360, nlat=181,nclusters=12):
        """ Necessary kwargs
            - patch_size, channles from FLAGS
            - nlons, nlats by user given
        """
        if os.path.exists(os.path.join(outputdir, filename)):
          os.remove(os.path.join(outputdir, filename))
        ## NetCDF writer
        # Define root group
        rootgrp = nc.Dataset(os.path.join(outputdir, filename), 'w', format='NETCDF4')

        # define Dimensions
        npatches    = rootgrp.createDimension("npatches",None)
        patchsize   = rootgrp.createDimension("patchsize",patch_size)
        channels    = rootgrp.createDimension("channels",channels)
        coordinate  = rootgrp.createDimension("coordinate",None)
        latlon      = rootgrp.createDimension("latlon",2)
        Latitude    = rootgrp.createDimension("Latitude", nlat)
        Longitude   = rootgrp.createDimension("Longitude",nlon)

        # define metadata
        rootgrp.hdfname    = obj['filename']
        rootgrp.instrument = obj['instrument']
        rootgrp.expname    = obj['model']['expname']
        rootgrp.layer_name = obj['model']['layer_name']

        #TODO: if you attach patch data, comment off patches in following
        # create and assign variables in root group
        # - patch
        #patches_ = rootgrp.createVariable('patches', 'f4', 
        #              ('npatches', 'patchsize', 'patchsize', 'channels',))
        #patches_.description = "ocean only patch data with >30% cloudy pixel QC"
        #patches_[:,:,:,:] = obj['patches'].astype(np.float32)
        # - coordinate
        coords_array = np.concatenate( 
            list(map(lambda x: np.expand_dims(np.asarray(x),axis=0), obj['coordinate']) ) 
        )
        coords_ = rootgrp.createVariable('coords', 'f4', ('coordinate','latlon',))
        coords_[:,:] = coords_array.astype(np.float32)
        
        # - label
        labels_ = rootgrp.createVariable('labels', 'i4', ('npatches',))
        labels_ = rootgrp.createVariable(f"labels_nc-{nclusters}", 'i4', ('npatches',))
        labels_[:] = obj['label'] 

        # coordination for aggregation data
        lats = rootgrp.createVariable('Latitude', 'f4', ('Latitude',))
        lons = rootgrp.createVariable('Longitude', 'f4', ('Longitude',))
        lats[:] =  np.arange(nlat).astype(np.float32)
        lons[:] =  np.arange(nlon).astype(np.float32)
        

        # *aggregation data*
        # - label_aggregation
        label_aggr_array_ = rootgrp.createVariable(f'nc-{nclusters}-label_1deg_aggregation', 'f4', 
                                ('Latitude', 'Longitude',))
        label_aggr = _label_aggr_save_helper(obj['label_aggregation'], nlat=nlat,nlon=nlon)
        label_aggr_array_[:,:] = label_aggr.astype(np.float32)

        # - param_aggregation
        map_array_dict = _param_aggr_save_helper(obj['param_aggregation'], nlat=nlat, nlon=nlon )
        for idx, (key, param_aggr_array) in  enumerate(map_array_dict.items() ):
            tmp = rootgrp.createVariable(f'{key}_1deg_aggregation', 'f4', ('Latitude', 'Longitude',))
            tmp[:,:] = param_aggr_array.astype(np.float32)


        # subgroup : patch-level mean and std
        statsgrp1 = rootgrp.createGroup('/statistics/mean')
        statsgrp2 = rootgrp.createGroup('/statistics/std')
        statsgrp3 = rootgrp.createGroup('/statistics/category')
        for key in map_array_dict.keys():
            if key == 'Cloud_Phase_Infrared_1km':
                rootgrp.createVariable(f'/statistics/category/mode', 'i4', ('npatches',))
                rootgrp.createVariable(f'/statistics/category/clear-sky', 'i4', ('npatches',))
                rootgrp.createVariable(f'/statistics/category/liquid', 'i4', ('npatches',))
                rootgrp.createVariable(f'/statistics/category/ice', 'i4', ('npatches',))
                rootgrp.createVariable(f'/statistics/category/undef', 'i4', ('npatches',))
                cph_modes = [] 
                skys = [] 
                liquids = []
                ices = []
                undefs = []
                for istats  in obj['statistics'] :
                    cph_modes.append(istats[key]['mode']) 
                    skys.append(istats[key]['clear-sky']) 
                    liquids.append(istats[key]['liquid']) 
                    ices.append(istats[key]['ice']) 
                    undefs.append(istats[key]['undef']) 
                statsgrp3['mode'][:]      = np.asarray(cph_modes).astype(np.int32)
                statsgrp3['clear-sky'][:] = np.asarray(skys).astype(np.int32)
                statsgrp3['liquid'][:]    = np.asarray(liquids).astype(np.int32)
                statsgrp3['ice'][:]       = np.asarray(ices).astype(np.int32)
                statsgrp3['undef'][:]     = np.asarray(undefs).astype(np.int32)

            else:
                rootgrp.createVariable(f'/statistics/mean/{key}', 'f4', ('npatches',))
                rootgrp.createVariable(f'/statistics/std/{key}', 'f4', ('npatches',))
                tmp_means = []
                tmp_stdvs = []
                for  istats in obj['statistics']:
                  tmp_means.append(istats[key]['mean'])
                  tmp_stdvs.append(istats[key]['std'])
                statsgrp1[key][:] = np.asarray(tmp_means).astype(np.float32)
                statsgrp2[key][:] = np.asarray(tmp_stdvs).astype(np.float32)
        # close file
        rootgrp.close()    


    def append_as_netcdf(obj, outputdir,filename, patch_size=128, channels=6, nlon=360, nlat=181,nclusters=12):
        """ Necessary kwargs
            - patch_size, channles from FLAGS
            - nlons, nlats by user given
        """
        ## NetCDF writer
        # Read root group
        rootgrp = nc.Dataset(os.path.join(outputdir, filename), 'a', format='NETCDF4')

        # *aggregation data*
        # - label_aggregation
        duplicate = False
        try: 
            label_aggr_array_ = rootgrp.createVariable(f'nc-{nclusters}-label_1deg_aggregation', 'f4', 
                                        ('Latitude', 'Longitude',))
        except Exception as e:
            duplicate = True
            pass

        label_aggr = _label_aggr_save_helper(obj['label_aggregation'], nlat=nlat,nlon=nlon)
        if duplicate:
            rootgrp.variables[f'nc-{nclusters}-label_1deg_aggregation'][:] = label_aggr.astype(np.float32)
        else:
            label_aggr_array_[:,:] = label_aggr.astype(np.float32)


        # label data per number of clusters 
        duplicate = False
        try:
            labels_ = rootgrp.createVariable(f"labels_nc-{nclusters}", 'i4', ('npatches',))
        except Exception as e:
            duplicate =  True
            pass

        if duplicate:
            rootgrp.variables[f'labels_nc-{nclusters}'][:] =   obj[f'label'] 
        else:
            labels_[:] = obj[f'label'] 

        # close file
        rootgrp.close()    

    aggr_label_dict = None
    aggr_param_dict = None
    none_time = time.time()
    if labels is not None:
      if isAppend:
        aggr_dict = aggregate_operator(labels, glats, glons, reso=1.0)
        aggr_label_dict = aggregate_label_counter(aggr_dict) # aggregation
        aggr_label_dict = label_retreive_fn(aggr_label_dict) # aggr + map

        # time store label operation
        timers.append(time.time())

      else:
        aggr_dict = aggregate_operator(labels, glats, glons, reso=1.0)
        aggr_label_dict = aggregate_label_counter(aggr_dict) # aggregation
        aggr_label_dict = label_retreive_fn(aggr_label_dict) # aggr + map

        # time store label operation
        timers.append(time.time())
        #physical parameter aggregation
        aggr_param_dict = {}
        for key  in param_keys:
          iparam = param_dict[key]
          aggr_dict  = param_groupby_operator(iparam, key, lats, lons) # group 1 deg scale
          if aggr_dict:
              map_aggr_dict = value_latlon_mapper(aggr_dict)
              aggr_param_dict[key]  = map_aggr_dict  # need to use `value_latlon_mapper` to use in analysis
          else:
              aggr_param_dict[key]  = {}

    else:
      timers.append(none_time)

    # time store param operation
    timers.append(time.time())
    if isAppend:
        if aggr_label_dict is not None:
            odict = OrderedDict()
            odict['label']        = labels
            odict['label_aggregation']  = aggr_label_dict 
            # save metadata as pkl
            filename = 'CMPST'+os.path.basename(_fname).strip('.hdf')+'.nc'
            append_as_netcdf(odict, outputdir,filename, patch_size=patch_size, channels=channels, nlon=nlon, nlat=nlat,nclusters=nclusters )
    else:
        if aggr_label_dict is not None and aggr_param_dict is not None :
            # collect metadata as OrderedDict
            odict = OrderedDict()
            odict['patches']      = patches  #.astype(np.float32) # reduce floatpoint
            odict['filename']     = _fname
            odict['coordinate']   = coords
            odict['latitude']     = glats
            odict['longitude']    = glons
            odict['instrument']   = instrument
            odict['label']        = labels
            odict['statistics']   = stats
            odict['model']        = {'expname': expname,'layer_name':layer_name }
            odict['label_aggregation']  = aggr_label_dict 
            odict['param_aggregation']  = aggr_param_dict 
            
            # save metadata as pkl
            filename = 'CMPST'+os.path.basename(_fname).strip('.hdf')+'.nc'
            save_as_netcdf(odict, outputdir,filename, patch_size=patch_size, channels=channels, nlon=nlon, nlat=nlat,nclusters=nclusters )
    # time end program
    timers.append(time.time())
    return timers


def parse_args(verbose=False):
    """
    workers_per_node: Number of workers started per node, 
                      which corresponds to the number of tasks 
                      that can execute concurrently on a node.
    nodes_per_block: Number of nodes requested per block
    """
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--EXEC",                type=str,  default='midway',help='selection of executor')
    p.add_argument("--queue",               type=str,  default='debug-cache-quad',help='selection of partition')
    p.add_argument("--walltime",            type=str,  default='01:00:00',help='computing time')
    p.add_argument("--stride",              type=int,  default=128,help='number of cores per node')
    p.add_argument("--patch_size",          type=int,  default=128,help='number of cores per node')
    p.add_argument("--nclusters",           type=int,  default=12,help='number of clusters')
    p.add_argument("--num_cores",           type=int,  default=1,help='number of cores per node')
    p.add_argument("--max_workers",         type=int,  default=1,help='max number of workers per node')
    p.add_argument("--cores_per_worker",    type=float,default=2.0, help='number of cores per work i.e. multi-threading')
    p.add_argument("--nodes_per_block",     type=int,  default=1,help='number of nodes requested per block')
    p.add_argument("--tf_datadir",          type=str,  default="./",help='tf mean and standard deviation data')
    p.add_argument("--instrument",          type=str,  default="terra",help='choose terra or aqua')
    p.add_argument("--hdf_mod02_datadir",   type=str,  default="./",help='mod02 hdf data directory')
    p.add_argument("--hdf_mod03_datadir",   type=str,  default="./",help='mod03 hdf data directory')
    p.add_argument("--hdf_mod06_datadir",   type=str,  default="./",help='mod06 hdf data directory')
    p.add_argument("--outputbasedir",       type=str,  default="./",help='output data directory')
    p.add_argument("--centroids_filename",  type=str,  default="./",help='centroids data')
    p.add_argument("--model_datadir",       type=str,  default="./",help='trained autoencoder directory')
    p.add_argument("--layer_name",          type=str,  default="./",help='trained autoencoder directory')
    p.add_argument("--expname",             type=str,  default="./",help='trained autoencoder directory')
    # new in scale
    p.add_argument("--date",                type=int,  default=1,help='date of first day to run exp')
    p.add_argument("--ndays",               type=int,  default=4,help='number of days to run from FLAGS.date')
    p.add_argument("--ocean_only",          action='store_true', help='patches on ocean only; then attach this argument')
    p.add_argument("--append",              action='store_true', help='write-add clustering results to existing saved pkl file; then attach this argument')
    FLAGS = p.parse_args()
    if verbose:
        for f in FLAGS.__dict__:
            print("\t", f, (25 - len(f)) * " ", FLAGS.__dict__[f])
        print("\n")
    return FLAGS

###############################################################################################
  
# Argparse PARAM
FLAGS=parse_args()

# FIX PARAM
init_blocks = 1
min_blocks = 1
max_blocks = 1

if FLAGS.EXEC == 'local':
    config = Config(
                executors=[
                    HighThroughputExecutor(
                        label="htex_Local",
                        worker_debug=True,
                        cores_per_worker=FLAGS.num_cores,
                        provider=LocalProvider(
                            channel=LocalChannel(),
                            init_blocks=1,
                            max_blocks=1,
                            #worker_init='source activate parsl_py36',
                        ),
                    )
                ],
                strategy=None,
            )
elif FLAGS.EXEC == 'theta':
    config = Config(
        executors=[
            HighThroughputExecutor(
                label='theta_local_htex_multinode',
                max_workers=FLAGS.max_workers,  # argument
                cores_per_worker=FLAGS.cores_per_worker,
                address=address_by_hostname(),
                provider=CobaltProvider(
                    #queue='default',
                    #queue='debug-cache-quad',
                    queue=FLAGS.queue,
                    account='Clouds',
                    launcher=AprunLauncher(overrides=""),
                    #walltime='00:30:00',
                    #walltime='01:00:00',
                    walltime=FLAGS.walltime,
                    nodes_per_block=FLAGS.nodes_per_block,
                    init_blocks=init_blocks,
                    min_blocks=min_blocks,
                    max_blocks=max_blocks,
                    # string to prepend to #COBALT blocks in the submit
                    # script to the scheduler eg: '#COBALT -t 50'
                    #scheduler_options='module load miniconda-3; conda activate tf2-gpu',
                    scheduler_options='#COBALT --attrs filesystems=home,eagle',
                    # Command to be run before starting a worker, such as:
                    # 'module load Anaconda; source activate parsl_env'.
                    worker_init='module load miniconda-3; conda activate tf2-gpu-py3.8.3',
                    cmd_timeout=120,
                ),
            )
        ],
        #monitoring=MonitoringHub(
        #   hub_address=address_by_hostname(),
        #   hub_port=6553,
        #   resource_monitoring_interval=10,
        #)
    )

elif FLAGS.EXEC == 'midway':
   config = Config(
       executors=[
           HighThroughputExecutor(
               label='Midway_HTEX_multinode',
               worker_debug=False,
               cores_per_worker=FLAGS.cores_per_worker,
               address=address_by_hostname(),
               #max_workers=FLAGS.num_cores,
               provider=SlurmProvider(
                   FLAGS.queue,
                   launcher=SrunLauncher(),
                   nodes_per_block=FLAGS.nodes_per_block,
                   cores_per_node=FLAGS.num_cores,
                   init_blocks=init_blocks,
                   min_blocks=min_blocks,
                   max_blocks=max_blocks,
                   exclusive=False,
                   walltime=FLAGS.walltime,
                   worker_init='module load python ; source activate tf2-gpu',
                   #worker_init='module load python ; source activate tf2-gpu',
                    #scheduler_options='#SBATCH --qos=broadwl',
                    #worker_init='source activate tf2-gpu',
                    #walltime='04:00:00'
                ),
            )
        ],
       monitoring=MonitoringHub(
           hub_address=address_by_hostname(),
           hub_port=6553,
           resource_monitoring_interval=10,
       )
    )
#elif xxxx:
# add your custom config
else:
    pass

# Load config
parsl.load(config)

timers = []
# loop to execute the simulation app
s = time.time()
for idate in range(FLAGS.date, FLAGS.date+FLAGS.ndays+1, 1): 
    ctimestamp = str(idate).zfill(3)

    # Make output directory
    outputdir = f'{FLAGS.outputbasedir}/{ctimestamp}/nc{FLAGS.num_cores}-npb{FLAGS.nodes_per_block}-cpw{FLAGS.cores_per_worker}-ib{init_blocks}-mb{min_blocks}-mb{max_blocks}'
    os.makedirs(outputdir,exist_ok=True)


    if FLAGS.instrument.lower() == 'terra':
        prefix02="MOD021KM"
    elif FLAGS.instrument.lower() == 'aqua':
        prefix02="MYD021KM"

    # create filelist from all modis files in subdirectories
    filelist=[]
    for idx, (dirpaths, dirnames, filenames) in enumerate(os.walk( os.path.join(FLAGS.hdf_mod02_datadir,ctimestamp ) )): 
        if not dirnames:
            filelist.extend(glob.glob(os.path.join(dirpaths,f"{prefix02}*.hdf"))) 

    if FLAGS.instrument.lower() == 'terra':
        prefix03  ="MOD03.A"
        prefix06  ="MOD06_L2.A"
        product   ='mod02'
        timerfile ='MOD2003_eachTimer'
    else:
        prefix03  ="MYD03.A"
        prefix06  ="MYD06_L2.A"
        product   ='myd02'
        timerfile ='MYD2003_eachTimer'


    for ifile in filelist:
        patches = patch_creation(filepath=os.path.join(FLAGS.hdf_mod02_datadir,ctimestamp), 
                        filenames=ifile,
                        instrument=FLAGS.instrument,
                        ems_band=["20","28","29",'31'],
                        stride=FLAGS.stride, patch_size=FLAGS.patch_size, channels=6,
                        mod03_datadir=os.path.join(FLAGS.hdf_mod03_datadir,os.path.join(*os.path.dirname(ifile).split('/')[-2:]) ),prefix03=prefix03,
                        mod06_datadir=os.path.join(FLAGS.hdf_mod06_datadir,os.path.join(*os.path.dirname(ifile).split('/')[-2:]) ),prefix06=prefix06,
                        outputdir=outputdir,
                        thres_ocean_frac=0.999,thres_cloud_frac=0.3,
                        ocean_only=FLAGS.ocean_only,
                        global_mean_file=os.path.join(FLAGS.tf_datadir,f'{product}_ocean_band28_29_31_gmean.npy'), 
                        global_stdv_file=os.path.join(FLAGS.tf_datadir,f'{product}_ocean_band28_29_31_gstdv.npy'),
                        centroids_filename=FLAGS.centroids_filename,
                        model_datadir=FLAGS.model_datadir,
                        expname=FLAGS.expname,
                        layer_name=FLAGS.layer_name,
                        nclusters=FLAGS.nclusters,
                        nlat=181, nlon=360,
                        isAppend=FLAGS.append,
        )
        timers.append(patches)

all_timers = [ t.result() for t in timers]
np.save(os.path.join(outputdir,timerfile), all_timers)
print('normal end patch_creation')

elapsed_time = time.time() - s
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
