## Work directory to open/decode hdf data and analyze them
----------

### Programs

- 000_example_open-decode.ipynb
  Notebook to practice how to open/decode/pre-process hdf data.

- prg_example_open-decode.ipynb  
  Python program to run entire open/decode/pre-process. Content is same as `000_example_open-decode.ipynb`.


### Useful resource

- User Guide for MODIS021KM (MUST READ Chapter.5)  
  [User Guide MODIS L1B: MOD021KM](https://mcst.gsfc.nasa.gov/content/l1b-documents)

- Website to learn an example of hdf file operation especially targets MODIS product  
  [How to read a MODIS HDF4 file using python and pyhdf ?](https://www.science-emergence.com/Articles/How-to-read-a-MODIS-HDF-file-using-python-/)



### Process Patches with Decoding HDF File

#### Step 0
- Read `Useful resource`

#### Step 1
- Practice how to open/decode/treat MODIS HDF data (MOD021KM and MOD35_L2)

#### Step 2
- Download necessary dataset to your storage
- Based on `clustering_invalid_filelist.txt`, make directories for these MOD02 data
- Run `job_StatsInvPixel.bash` which implemets `prg_StatsInvPixel.py`  
  Here, mod02_dir should be specified where you moved selected MOD02 data by `filelist_with_invalids.txt`

#### Step 3
- Open output npz file. The numpy array contain  
    `filename` : mod02 filename having invalid pixel   
    `pixel_list` :  list of number of invalid pixel in each patch  
    `patch_list` :  list of 1. Each 1 entry means there is 1 patch. np.sum(patch_list) means number of patches having more than 30% of clouds

#### Step 4
- Plot graph, X axis is number of invalid pixel. Y axis is number of patches.

----------

# find_invalids.py

### Katy Koenig

The code in this file is used to check hdf files for invalid pixels.

Please note that relevant mod02 and mod03 files should be downloaded prior to running in this py file.

## Getting Started

### Necessary Modules

* numpy 1.16.4
* pandas 0.24.2
* pyhdf 0.10.1

## How to Use

Edit and run the revelant bash scripts below.

1. [fi_get_invalid_info.bash](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/work_hdfs/fi_get_invalid_info.bash)
  * This bash script is used for hdf files that are downloaded via the combined repo, i.e. files are of the entire global and not location specific.
  * This bash script creates a csv with each row as a patch. Columns in the csv are 'filename', 'patch_no', 'inval_pixels', where 'filename' refers to the MOD02 filename.

2. [fi_get_info_for_location.bash](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/work_hdfs/fi_get_info_for_location.bash)
  * This bash script is used for hdf files that are downloaded via the api-requests, i.e. the files are of a specific location.
  * This bash script creates a csv with each row as a patch. Columns in the csv are 'filename', 'patch_no', 'invalid_pixels', 'geometry', where 'filename' refers to the location and date of file.


# geolocation.py

### Katy Koenig

## Getting Started

The code in this file is used to connect MOD02, MOD35 and MOD03 files to create a csv with each row as a patch. The columns in the saved csv are 'filename', 'patch_no', 'invalid_pixels', and 'geometry', where 'filename' is the MOD02 filename.

### Necessary Modules

* numpy 1.16.4
* pandas 0.24.2
* geopandas 0.5.1
* matplotlib 3.1.1
* pyproj 1.9.5.1
* shapely 1.6.4.post2

## How to Use

1. Edit and run [geo.bash](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/work_hdfs/geo.bash)
* Note: the input csv should be a list of MOD02 filenames (each with its own row) and with the header/first row as "filename". A zipped example can be found [here](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/work_hdfs/mod02_geo_example.zip)


# map_cluster.py

### Katy Koenig

## Getting Started

The code in this file analyzes cluster iterations (e.g. 80,000 patches clustered into 20 groups) to see the relationships between location, date and group number.

### Necessary Modules

* numpy 1.16.4
* pandas 0.24.2
* geopandas 0.5.1
* matplotlib 3.1.1

## How to Use

1. Edit & run [map_clusters.bash](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/work_hdfs/map_clusters.bash)
 * This bash script creates and saves the following:
   1. A csv with each row being a patch. The columns in the csv are 'file', 'indices', 'cluster_num', 'geom'.
   2. Two images of this csv data:
      * An image with a map for each date in the dataframe where the colors relate to the cluster_col (cluster number). The colors are each cluster number are invariant across each plot.
      * An image with a map for each cluster in the dataset.

 * Please note that MOD03 hdf files that correlate MOD02/MOD35 hdf files used in the clustering process must already be downloaded. I would recommend using [download_spec_datetimes.bash](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/combined/download_spec_datetimes.bash) to download only the specific date/time MOD03 files needed.

2. Additional Functions
	1. map_all_discrete
 		* This function creates and saves one map in which each unique value of the column specified is a different color.
 	2. map_all_continuous
 		* This function creates and saves one map in which the color represents the continuous values of the specified column.
