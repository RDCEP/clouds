# Code for Labeling Cloud Data and Clustering Model Validation

### Ruby Werman

## Getting Started
1. cloud_labeling.ipynb
	* Jupyter Notebook file to label and output patch data from a given date range
2. visualize_patches.ipynb
	* Jupyter Notebook file to cluster labeled patch data, create visualizations, and remove poorly labeled patches

### Necessary Modules:

* matplotlib
* os
* sys
* glob
* numpy
* matplotlib
* pyhdf.SD
* Tensorflow 1.12.0 for CPU
* pandas
* seaborn
* math
* sklearn

## How to Label Data

1. Necesary elelments:
  * lib_hdfs directory
  * .txt file of dates (see clouds/src_analysis/dates for examples)
  * MOD02, MOD35 data from the NASA LAADS website (see [here](https://github.com/RDCEP/clouds/tree/mod021KM/src_analysis/combined) for download instructions)

2. Run cloud_labeling.ipynb

3. After running the notebook (and labeling), you will have the necessary files for clustering and validation. For each date that you labeled patches for, you will have the following:
  * close_cells_coords_date.npy (patch coordinate data for closed cell patches)
  * open_cells_coords_date.npy (patch coordinate data for open cell patches)
  * close_cells_mod02_date.npy (MOD02 data for closed cell patches)
  * open_cells_mod02_date.npy (MOD02 data for open cell patches)
  * close_35_date.npy (MOD35 data for closed cell patches)
  * open_35_date.npy (MOD35 data for open cell patches)
  * open_dates.txt (date file for all open cell patches)
  * closed_dates.txt (date file for all close cell patches)
  
## How to Vizualize Data

1. Necessary elements:
  * lib_hdfs directory
  * encoder directory (see "load model" section of visualize_patches.ipynb)
  * The files listed under #3 above 
  
2. Run visualize_patches.ipynb
  * Edit num_clusters to change the number of clusters
  * Save plot images if desired

