### Clouds Project
Center for Robust Decisionmaking on Climate and Energy Policy (RDCEP)


---
#### Project Organization

- [Static](https://github.com/RDCEP/clouds/blob/master/static/reference.markdown):
All environment setup for UChicago and ANL clusters
- Run: Collections of jobs used to preprocess different sources of data,
and also launch training routines in different environments.
Instructions for running jobs on
[ANL/ALCF](https://github.com/RDCEP/clouds/blob/master/run/theta/ALCF_Running_ML_jobs.pdf).
- Reproduction: Codebase called by jobs for data translation, modelling
and analysis
- Output: Trained models outputs in TensorFlow format (HDF container
with weights; JSON file with execution graph definition)
- Logs: Execution logs of job runs
- Experimental: Set of very unstable prototype codebase of analysis and
functions considered to be used in the project
- Analysis: Set of jupyter notebooks with different techniques for
analysis of datasets, their encodings, as well on
unsupervised classification of latent representation.
- GEE: Routines developed using Google Earth Engine.

---
#### Main framework change
Create new branch ```mod021KM``` where modified codes for MOS021KM dataset are pushed.

- Reproduction/train.py  
  - add new parser arguments  
  - add tf.cast line to map tf.float64 to tf.float32 before tensorflow run
- Reproduction/pipeline/load.py  
  master    `tf.decode(data, tf.float32)`  
  mod021KM  `tf.decode(data, tf.float64)`  
  This modification means hdf to tfrecord pre-process was stored in 64bit data.  
- Preprocess code is modified
