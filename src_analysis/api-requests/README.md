# Code for Ordering & Downloading NASA MODUS HDF Files through NASA LWS API
### Katy Koenig

## Getting Started

This file contains the following:

1. api_request.py
	* python file containing functions for ordering and releasing hdf files through NASA api as well as wrapper function to order, download and release hdf files.
2. laads_data_download.py
	* python file to download ordered and processed files. Edited from the original version available provided by NASA here: https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/
3. coords.csv
	* csv file with coordinates of desired location of patches. Please note that this csv can be created and edited through the write_csv() function provided in api_request.py
4. label1.txt
	* txt file of desired dates for which to download data
5. total_bad_dates.csv
	* csv file with 
6. OpenClosedArea.txt
	* txt file linking the coordinates with location names
	* created using get_bad_dates() function in prg_gen_rndm_metadata.py located in clouds/src_analysis/combined

### Necessary Modules:

* requests 2.22.0
* bs4 0.0.1
* pandas 0.24.2

## How to Download Files

Note: you must already have a registered email address through NASA's LAADS DAAC website as well as an app_key created after registration.

### Through Command Line

1. Ensure you have the desired coordinates in coords.csv and desired dates in label1.txt. Edit files accordingly if desired.
2. Run the following in the command line:

`python3 api_request.py 'YOUREMAILADDRESS' 'YOURAPPKEY' 'DESIREDPRODUCTS'`

Notes:
	* 'YOUREMAILADDRESS' should be a string with the email address the user has registered with NASA.
	* 'YOURAPPKEY' should be a string with the app key the user has received from NASA.
	* 'DESIREDPRODUCTS' should be a string of the desired NASA products you wish to download, i.e. 'MOD35_L2'. If you would like to download multiple products please list them according to the following example: 'MOD35_L2,MYD021KM'

### Through Python3
1. Edit the api_request.py file to include user's NASA-registered email address and app key. This is located that the top of the file and noted by a comment for clarity.
2. Ensure you have the desired coordinates in coords.csv. As currently pushed on 27 June 2019, write_csv() generates a csv with coordinates for eight locations. If you would like to add to or delete from this list, please edit the code accordingly with corresponding cardinal directions and call write_csv().
3. Call combining_fn() from api_request.py. Notes:
	* Any available past orders for the user's account will be released when combining_fn() called and will have to be reorder if needed in the future.
	* Files will automatically be downloed into a folder called "hdf_files" in the user's current working directory. If the user would like to change the output file, please edit the OUTPUT_FILE string at the top of the api_request.pys
	* All orders placed with this call will be released upon termination of downloading processes. If process is interrupted, files ordered but not yet downloaded will not be released. If you wish to release all files then, please call clear_all_orders()
