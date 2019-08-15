# Code for Clustering Model Validation with the "model" patches and the 80k patches in the dataset

### Ruby Werman

## Getting Started
1. cluster_for_rcc.py
	* Python file to run the 80k patches through the autoencoder, combine feature data with "model" patches feature data, and cluster 
2. ruby_job.bash
	* Bash script to run `cluster_for_rcc.py` on Midway and save output file to one's computer
3. ruby_features.npy
	* Result of running 31 "model" patches through autoencoder (m01_b)
4. normed_ruby.npy
	* Result of running 31 "model" patches through normed autoencoder (m2_02_normed)
5. output_files
	* Directory containing the output files of running the bash script
		* `aggl_p-80_nc-20_m2_02_normed_train_patches_labels_2000-2018_random_aggl.npy` contains label data to then be 			used to analyze clusters
		* `80_nc-20_m2_02_normed_train_patches_labels_2000-2018_random_aggl.npy` contains the output of running TSNE 			on the patches for use in creating a TSNE visualization
		
	*You can use [`80k_with_31_patches_clustered.ipynb`](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/cloud_label/80k_with_31_patches_clustered.ipynb) to make these visualizations*
### Necessary Modules:

* os
* argparse
* glob
* time
* numpy
* random
* sys
* tensorflow
* sci-kit learn
