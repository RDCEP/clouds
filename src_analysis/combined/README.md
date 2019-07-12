# Code for Downloading Non-Location Specific Data via NASA Website

### Katy Koenig

## Getting Started
1. download.bash
	* bash script to download hdf files
2. prg_gen_rndm_metadata.py
	* py file that generates random, valid dates between two given dates
3. prg_geturl2.py
	* py file that with main driver functions to download data either between two given dates or all data for dates in a given txt file
4. clustering_mod35_list.txt
	* txt file with desired dates for downloads (previously downloaded by Katy)
5. no-data-dates.txt
	* txt file with list of dates that should not be downloaded

### Necessary Modules:

* requests 2.22.0
* beautifulsoup4 4.7.1
* html5lib 1.0.1
* pandas 0.24.2

## How to Download Files

1. Edit download.bash (or copy, rename and edit download.bash)
 There are two ways in which to download the edit and editing the bash script depends on your desired call:
	1. You wish to generate random dates to download.
	 	You must change the following arguments:
		* days (int): number of days you wish to randomly generate
		* start(str in the style of YYYY-MM-DD): lower bound for date generation, i.e. you cannot generate a date earlier than this date
		* end(str in the style of YYYY-MM-DD): lower bound for date generation, i.e. you cannot generate a date later than this date
	 2. You have a list of dates to download. <br />
	 	If you have a txt file of the dates for which you wish to download data (example: clustering_mod35_list.txt in this repo), you need only modify the `datedata` argument in the bash script with the path to the txt file as a string.

	 Note that for both options you **must** check (and probably edit) the following arguments:
	 1. outputdir
	 	* this is a string representing the directory in which you want to store your files (can be created in downloading process)
	 2. url
	 	* url as determined by desired product as a string
	 	* e.g. 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD06_L2/', 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD35_L2'
	 3. keyward
	 	* keyword associated with product as a string
	 	* e.g. 'MOD35_L2.A', 'MOD06_L2.A'
	 4. processors
	 	* number of desired processors to use as an integer
	 	* this program supports multiprocessing to increase speed
	 	* it is recommended to use no more than 1 less than your computer's CPU count

2. Run `bash download.bash`
