### Instructions for analysis
- Midway

Use sinteractive mode with high memory allocation.

Once running, get IP address via:

```/sbin/ip route get 8.8.8.8 | awk '{print $NF;exit}'```

Then launch Jupyter notebook:

