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

3. After running the notebook (and labeling), you will have the necessary file for clustering and validation. This file contains a list of labeled patch instances with the necessary information for the clustering model and analysis. 
  * patches_DDMMYYYY.npy, where DDMMYYYY is the date the patches were labeled
  
## How to Vizualize Data

1. Necessary elements:
  * lib_hdfs directory
  * encoder directory (see "load model" section of visualize_patches.ipynb)
  * patches_DDMMYYY.npy
  
2. Run visualize_patches.ipynb
  * Edit num_clusters to change the number of clusters for agglomerative clustering
  * Remove ambigious/mislabeled patches from patch list if necessary
  * Save plot images if desired

