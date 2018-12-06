import sklearn
import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

sys.path.insert(1, os.path.join(sys.path[0], ".."))
sys.path.append('../../')
from reproduction.pipeline.load import load_data
from reproduction import analysis

import multiprocessing as mp
from itertools import product

# Disable Warnings
import warnings; warnings.simplefilter('ignore')
tf.logging.set_verbosity(tf.logging.WARN)

def gridsearch(start, step, stop, max_samples=5000, sample_steps=4, trials=30):
    with open(ENCODER_DEF,"r") as f:
            encoder = tf.keras.models.model_from_json(f.read())
    encoder.load_weights(ENCODER_WEIGHTS)
    
    samples = np.logspace(np.log10(start+2), np.log10(max_samples), num=sample_steps).astype(int)

    def trial_test(i, j):
        search_results = []  # Force initialization
        print('Samples: ', i, ' Clusters: ', j)
        minfoac = []
        for trial in range(trials):
            data = analysis.AEData(load_data(DATA, encoder.input_shape[1:]), n=i)
            data.add_encoder(encoder)
            N_CLUSTERS = j
            ag1 = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit_predict(data.encs[:int(i / 2)])
            ag2 = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit_predict(data.encs)
            minfoac.append(sklearn.metrics.adjusted_mutual_info_score(ag1, ag2[:int(i / 2)]))
        minfo_mean = np.nanmean(minfoac)
        minfo_std = np.nanstd(minfoac)
        search_results.append((i, N_CLUSTERS, minfo_mean, minfo_std))
        print('Average Mutual information: ', minfo_mean, 'MI_STD: ', minfo_std, 'Precision: ',
              np.count_nonzero(~np.isnan(minfoac)), flush=True)
        return search_results

    print(list(product(samples, range(start, stop, step))), flush=True)
    with mp.Pool(processes=8) as pool:
        results = pool.starmap(trial_test, product(samples, range(start, stop, step)))

    return results
    

DATA = "/project/foster/clouds/data/2015_05/*.tfrecord"
ENCODER_DEF = "/home/rlourenco/rdcep_clouds/output/m9-22_oceans/encoder.json"
ENCODER_WEIGHTS = "/home/rlourenco/rdcep_clouds/output/m9-22_oceans/encoder.h5"


result = gridsearch(2,1,40,max_samples=10000, sample_steps=10)

df_export = pd.DataFrame.from_records(result, columns=['Samples','Clusters','Avg_MInfo','Std_MInfo'])
df_export.to_csv('searched_results.csv', encoding='utf-8', index=False)
