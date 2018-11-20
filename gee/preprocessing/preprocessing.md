### Google Earth Engine Preprocessing Routines
- mod09_export.js : Using MOD09 as base dataset, it exports a daily
mosaic per export execution in geotiff format for all seven reflectance
bands. Has option to limit extension to ocean limits, to export patches
that are above a certain cloud fraction. It performs cloud segmentation
using its cloud mask, with options of inputate missing values with a
flag or random noise.

- mod09_open_close_annotated_export.js : Quite the same as above, but
with a simplified interface in which the user may select geometries for
exports of open/close cell patterns. Needs refactoring for a more
generalized annotation export.