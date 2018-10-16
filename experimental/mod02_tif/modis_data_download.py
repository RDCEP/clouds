'''
Routine to download MOD021 HDF datasets.
Relies on modapsclient: https://github.com/chryss/modapsclient
'''
import modapsclient, os, posixpath, asyncio, logging, aiohttp
from contextlib import closing
from urllib.parse import urlsplit, unquote


def download(collection, # MODIS Collection
             initial_date, # Start date (inclusive)
             end_date, # End date (inclusive)
             north_lat=(90.0), # Upper bbox latitude
             south_lat=(-90.0), # Lower bbox latitude
             east_lon=(180.0), # Eastern bbox meridian
             west_lon=(-180.0), # Western bbox meridian
             collection_version=6 ,# MODIS collection version, 6 being the latest and default
             dest_folder='~/.',
             cpus=4):

    #TODO: Use argparse
    #TODO: option of saving in a custom folder
    #TODO: input info validation
    #TODO: use datetime objects for initial and end, cap to daily calls
    #Todo: Dump execution Log
    #TODO: Parallelize with PyMPI

    # Define client object
    a = modapsclient.ModapsClient()

    # Perform dataset search
    file_list = a.searchForFiles(collection, initial_date, end_date,
                                 north_lat, south_lat, east_lon,west_lon, collection=collection_version)
    print(file_list)

    # Test single file
    download_urls = []
    for fileid in file_list:
        transientobj = a.getFileUrls(fileid)
        download_urls = download_urls + transientobj
    print(download_urls)


    def url2filename(url):
        urlpath = urlsplit(url).path
        basename = posixpath.basename(unquote(urlpath))
        if (os.path.basename(basename) != basename or
                unquote(posixpath.basename(urlpath)) != basename):
            raise ValueError  # reject '%2f' or 'dir%5Cbasename.ext' on Windows
        return basename

    @asyncio.coroutine
    def download(url, session, semaphore, chunk_size=1 << 15):
        with (yield from semaphore):  # limit number of concurrent downloads
            filename = url2filename(url)
            logging.info('downloading %s', filename)
            response = yield from session.get(url)
            with closing(response), open(filename, 'wb') as file:
                while True:  # save file
                    chunk = yield from response.content.read(chunk_size)
                    if not chunk:
                        break
                    file.write(chunk)
            logging.info('done %s', filename)
        return filename, (response.status, tuple(response.headers.items()))

    urls = download_urls
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    with closing(asyncio.get_event_loop()) as loop, closing(aiohttp.ClientSession()) as session:
        semaphore = asyncio.Semaphore(16)
        download_tasks = (download(url, session, semaphore) for url in urls)
        result = loop.run_until_complete(asyncio.gather(*download_tasks))

    return


''' 
Example of execution:

download('MOD021KM', '2017-01-01', '2017-01-01', 90.0, -90.0, 180.0, -180.0, 6, '/', cpus=6 )
'''
