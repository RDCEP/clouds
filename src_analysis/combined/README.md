# Code for Downloading Non-Location Specific Data via NASA Website

### Katy Koenig

## Getting Started

This code is optimized to download non-location specific data through two different methods:

1. You have specific dates and times of data you want to download. For example, you have MOD02 data downloaded and want to download only the corresponding MOD03 data for that date time combination (instead of all times for specific dates). If this is you, use "download_spec_datetimes" as your template for downloading data.

2. You have no "base" data, but you do have a list of dates for which you want to download data OR two dates between which you want the data to fall. If this describes what you're trying to do, use "download_daterange.bash" as your example.


### Explanation of Files in this Directory:

1. download_daterange.bash
	* bash script to download hdf files between two dates or for a given list of dates
2. download_spec_datetimes.bash
	* bash script to download hdf files by specific dates and times
3. prg_gen_rndm_metadata.py
	* py file that generates random, valid dates between two given dates
4. prg_geturl2.py
	* py file that with main driver functions to download data either between two given dates or all data for dates in a given txt file
5. no-data-dates.txt
	* txt file with list of dates that should not be downloaded
6. clustering_mod35_list.txt
	* txt file example for "datedata" argument in download_daterange.bash
7. datetime_example.csv
	* example csv for download_spec_datetimes.bash "input_csv" argument


### Necessary Modules:

* requests 2.22.0
* beautifulsoup4 4.7.1
* html5lib 1.0.1
* pandas 0.24.2

## How to Download Files

1. Edit (or copy, rename and edit) the revelant bash script. Bash scripts are described in the "Getting Started" section above.

	1. [download_daterange.bash](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/combined/download_daterange.bash): There are two ways in which to download and edit the bash script depends on your desired call:
		1. You wish to generate random dates to download.
		 	You must change the following arguments:
			* days (int): number of days you wish to randomly generate
			* start (str in the style of YYYY-MM-DD): lower bound for date generation, i.e. you cannot generate a date earlier than this date
			* end (str in the style of YYYY-MM-DD): upper bound for date generation, i.e. you cannot generate a date later than this date
		 2. You have a list of dates to download. <br />
		 	If you have a txt file of the dates for which you wish to download data (example: [clustering_mod35_list.txt](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/combined/clustering_mod35_list.txt)), you need only modify the `datedata` argument in the bash script with the path to the txt file as a string.
	2. [download_spec_datetimes.bash](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/combined/download_spec_datetimes.bash)
		* Check to ensure the input_csv has the correct csv. An example of a correct input is [datetime_example.csv](https://github.com/RDCEP/clouds/blob/mod021KM/src_analysis/combined/datetime_example.csv).

	 Note that for both options you **must** check (and probably edit) the following arguments:
	 1. outputdir
	 	* this is a string representing the directory in which you want to store your files (can be created in downloading process)
	 2. keyword
	 	* keyword associated with product as a string
	 	* "MOD02", "MOD35", "MOD06" and "MOD03" are the **only** arguments currently allowed. If you would like to download a different product, you will need to add a key, value pairing of the produce and base url in the BASE_URL dictionary at the top of prg_geturl2.py.
	 3. processors
	 	* number of desired processors to use as an integer
	 	* this program supports multiprocessing to increase speed
	 	* it is recommended to use no more than 1 less than your computer's CPU count

2. Run `bash download_daterange.bash` or `bash download_spec_datetimes.bash`.
