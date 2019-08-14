## Library and program for pre-post processes

### Directory
- api-requests  
  directory to download MODIS product via API          
- combined  
  directory for combined data-download program + random dates generator program
- conda_envs    
  Takuya's frequently using conda environment(s)  
- dates  
  dates information for training, validation and clustering 
- lib_hdfs  
  directory for decoding hdf library.
  * Important notice  
    Major updated done on June 26th 2019. Check below modification was seen at following program(s)  
      
    Filename: alignment_lib.py  
    Function: mod02_proc_sds  

    Contents  
    ```
     # invalid value process
     # TODO: future change 32767 to other value
     invalid_idx = np.where( array > 32767 )
     if len(nan_idx) > 0:
         array[invalid_idx] = np.nan
     else:
         pass
    ``` 
- cloud_label
  directory containing notebooks used to label MODIS data, cluster, and create visualizations
- load_hdfs  
- metadata
- tools
