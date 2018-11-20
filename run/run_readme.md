### Instructions to run jobs

---
Two environments:
- UChicago RCC
    - Broadwl: CPU cluster
    - gpu2 : GPU cluster
- ANL ALCF
    - Cooley: GPU visualization cluster
    - Theta: CPU cluster

#### Running jobs on RCC
The current job definition is a very simplified one and limited to the
context of this codebase, not excluding reading of the official
[RCC documentation](). In our documentation jobs and
environment setups are taking in consideration Midway 2.

##### Data translation
Jobs of data translation consists on transforming geospatial containers
to TensorFlow Record format (TFRecord), only format able to be consumed
at the modelling routine.

We currently transform data from the following formats:
- MOD09: MODIS Surface Reflectance. This is a Level 2 product exported
from a Google Earth Engine routine. Files come in a GeoTIFF container
with just the seven reflectance bands. Preprocessing routines available
at [/gee]().



#### Running jobs on ALCF
TBD