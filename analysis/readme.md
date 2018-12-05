### Instructions for analysis
- Midway

Use sinteractive mode with high memory allocation
(16 GB RAM is ok for basic visualization;
32 is sufficient for a single GeoTIFF swath)


Once running, get IP address via:

```/sbin/ip route get 8.8.8.8 | awk '{print $NF;exit}'```

Then launch Jupyter notebook:

```jupyter notebook --no-browser --ip=<the ip address>```

This may be accessed within a Thinlic client or through an ssh tunnel