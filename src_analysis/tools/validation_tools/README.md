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
