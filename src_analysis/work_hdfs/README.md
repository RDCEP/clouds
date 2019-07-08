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
