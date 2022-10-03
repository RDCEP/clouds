## Clouds Project
![icon](docs/images/Clouds-Logo.png)  
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

Clouds project provides an unsupervised machine learning algorithm for automated clustering and pattern discovery in cloud imagery, 
and a dataset resulting from the algorithm that is applied to 22 years (2000-2021) of [Moderate Resolution Imaging Spectroradiometer (MODIS)](https://ladsweb.modaps.eosdis.nasa.gov) on NASA's Aqua and Terra satellites
to contribute to the democratization of climate research.


---------------------------
## Download AICCA dataset

You need to register [Globus](https://www.globus.org/data-transfer), a high-speed data transfer service, to download AICCA - patch and grid-cell dataset from following link:  
### AICCA - Patch dataset  
#### NetCDF version
Version.1 (Complete 2000 -- 2021)   
[AICCA_Patch](https://app.globus.org/file-manager?origin_id=dc1bfe8a-cbc9-11ec-b95a-0f43df60473d&origin_path=%2F)

#### CSV format (Complete 2000 -- 2021)
This CSV format version was supposed to a pre-stage for grid-cell dataset. But we are aware that the dataformat is easily used with Pandas and decided to publickly open this data.   
[AICCA_Patch_CSVformat](https://app.globus.org/file-manager?origin_id=86f8615c-32e7-11ed-ba3e-d5fb255a47cc&origin_path=%2F)  


### AICCA - Grid-cell dataset
Version.1 (Complete 2000 -- 2021)  
[AICCA Grid cell](https://app.globus.org/file-manager?origin_id=085d7744-ee7f-11ec-b3bd-15403b7b75ed&origin_path=%2F)

### AICCA: AI-driven Cloud Classification Atlas
A novel cloud classification dataset produced by applying modern unsupervised deep learning methods to identify robust and meaningful clusters of cloud patterns.
AICCA delivers in a compact form (tens of gigabytes of class labels, with high spatial and
temporal resolution) information currently accessible only as hundreds of terabytes of multi-spectral images.
AICCA enables data-driven diagnosis of patterns of cloud organization, provide insight into their evolution on
timescales of hours to decade

### Cite AICCA
If you use AICCA dataset for your work, please cite the paper and dataset:   
```
@article{kurihana2022aicca,
  title={AICCA: AI-driven Cloud Classification Atlas},
  author={Kurihana, Takuya and Moyer, Elisabeth and Foster, Ian},
  journal={arXiv preprint arXiv:2209.15096},
  year={2022}
}
```


---------------------------
##  Machine Learning Source Code
### RICC: Rotation-Invariant Cloud Clustering
A [rotation-invariant cloud clusering (RICC)](https://ieeexplore.ieee.org/document/9497325) is a data-driven unsupervised learning apporoach 
that leverages rotaion-invariant autoencoder and hierarchical agglomerative clustering to automate the clustering of cloud patterns and textures 
without any assumptions concerning artitifical cloud categories.   

If you find RICC is applicable and useful, please cite this paper:
```
@article{kurihana2021cloud,  
    title={Data-Driven Cloud Clustering via a Rotationally Invariant Autoencoder},   
    author={Kurihana, Takuya and Moyer, Elisabeth and Willett, Rebecca and Gilton, Davis and Foster, Ian},  
    journal={IEEE Transactions on Geoscience and Remote Sensing},   
    year={2022},  
    volume={60},   
    pages={1-25},  
    doi={10.1109/TGRS.2021.3098008}}
```

### Acknowledgments
This work is supported by the [AI for Science program of the Center for Data and Computing at the University of Chicago](https://datascience.uchicago.edu/research/is-climate-change-changing-clouds/) and
[Center for Robust Decisionmaking on Climate and Energy Policy (RDCEP)](http://www.rdcep.org/).  
We thank Argonne Leadership Computing Facility and University of Chicago’s Research Computing Center for access to computing resources.
