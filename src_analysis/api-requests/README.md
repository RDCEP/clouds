# Code for Ordering & Downloading NASA MODUS HDF Files through NASA LWS API
### Katy Koenig

## Getting Started

This repo contains the following:

1. api_request.py
	* python file containing functions for ordering and releasing hdf files through NASA api as well as wrapper function to order, download and release hdf files.
2. laads_data_download.py
	* python file to download ordered and processed files. Edited from the original version available provided by NASA here: https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/
3. example.bash
	* example bash file to order via the API
4. coords.csv
	* csv file with coordinates of desired location of patches. Please note that this csv can be created and edited through the write_csv() function provided in api_request.py
5. label1.txt
	* example txt file of desired dates for which to download data
6. total_bad_dates.csv
	* csv file with dates that should not be downloaded
7. OpenClosedArea.txt
	* txt file linking the coordinates with location names
	* created using get_bad_dates() function in prg_gen_rndm_metadata.py located in clouds/src_analysis/combined

### Necessary Modules:

* requests 2.22.0
* bs4 0.0.1
* pandas 0.24.2

## How to Download Files

Note: you must already have a registered email address through NASA's LAADS DAAC website as well as an app_key created after registration.

1. Edit (or copy, rename and edit) [example.bash](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/api-requests/example.bash). Here's a walkthrough of the arguments:
	1. email
		* This is the email address you have previously registered with NASA.
	2. app_key
		* This is the app key that corresponds to your NASA-registered email address. Note: you have to request an app key after making your NASA account. Please note that you can request multiple app keys.
	3. products
		* This is the product or list of products that you are requested. Examples in include 'MOD35_L2' or 'MOD35_L2,MYD021KM'. A full list of available products can be found [here](https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/listProducts?).
	4. date_file([example](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/api-requests/dates.txt))
		* txt file with desired dates to be downloaded.
	5. coords_file([example](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/api-requests/coords.csv))
	 	* csv file with the desired coordinates data. This can be created using the write_csv() function in the api_request.py.
	6. desired_dir
		* This is the base directory in which you would like your files saved. The program with create directories in this directory corresponding to locations and dates as well.
	7. addl_info
		* This argument is a boolean depending on if you would like your mosaic post-processing to your requested data.

2. Run `bash example.bash` in command line.

### Notes:
	* Any available past orders for the user's account will be released when bash called to free space for new orders and will have to be reorder if needed in the future.
	* All orders placed with this call will be released upon termination of downloading processes. If process is interrupted, files ordered but not yet downloaded will not be released. If you wish to release all files then, please call clear_all_orders() in api_requests.py.
