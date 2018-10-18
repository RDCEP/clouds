'''
Routine to download MOD021 HDF datasets.
Relies on modapsclient: https://github.com/chryss/modapsclient
'''
import modapsclient, os, posixpath, asyncio, logging, aiohttp, datetime
from contextlib import closing
from urllib.parse import urlsplit, unquote
import urllib.request
from mpi4py import MPI


def url2filename(url):
    urlpath = urlsplit(url).path
    basename = posixpath.basename(unquote(urlpath))
    if (os.path.basename(basename) != basename or
            unquote(posixpath.basename(urlpath)) != basename):
        raise ValueError
    return basename


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
    #Todo: Dump execution Log
    #TODO: Write function to retrieve ids, and retry in case of unavailability


    # Define client object
    a = modapsclient.ModapsClient()

    # Perform dataset search
    file_list = a.searchForFiles(collection, initial_date, end_date,
                                 north_lat, south_lat, east_lon,west_lon, collection=collection_version)
    # print(file_list)

    # Test single file
    print(file_list)
    download_urls = []
    unretrievable_ids = []
    for fileid in file_list:
        # Test if HDF URL is available
        fileprop = a.getFileProperties(fileid)
        print(fileprop)
        for item in fileprop:
            if item['online'] != 'true':
                transientobj = a.getFileUrls(fileid)
                download_urls = download_urls + transientobj
            else:
                print('FileId: ', fileid, ' is unavailable to retrieve URL')
                unretrievable_ids.append(fileid)

    print('WARNING: Following file ids were not retrieved', unretrievable_ids)
    # print(download_urls)



    # @asyncio.coroutine
    # def download(url, session, semaphore, chunk_size=1 << 15):
    #     with (yield from semaphore):  # limit number of concurrent downloads
    #         filename = url2filename(url)
    #         logging.info('downloading %s', dest_folder+filename)
    #         response = yield from session.get(url)
    #         with closing(response), open(dest_folder+filename, 'wb') as file:
    #             while True:  # save file
    #                 chunk = yield from response.content.read(chunk_size)
    #                 if not chunk:
    #                     break
    #                 file.write(chunk)
    #         logging.info('done %s', dest_folder+filename)
    #     return filename, (response.status, tuple(response.headers.items()))

    # urls = download_urls

    # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    # with closing(asyncio.get_event_loop()) as loop, closing(aiohttp.ClientSession()) as session:
    #     semaphore = asyncio.Semaphore(16)
    #     download_tasks = (download(url, session, semaphore) for url in urls)
    #     result = loop.run_until_complete(asyncio.gather(*download_tasks))

    for i in download_urls:
        filename = url2filename(i) # get filename
        # with open(dest_folder+filename, wb)as file:
        urllib.request.urlretrieve(i,os.path.join(dest_folder,filename))


    return

def date_list(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days+1)]
    # for date in date_generated:
        # print(date.strftime("%Y-%m-%d"))
    return date_generated

''' 
Example of execution:

download('MOD021KM', '2017-01-01', '2017-01-01', 90.0, -90.0, 180.0, -180.0, 6, '/', cpus=6 )
'''

# Large call
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


start_date = '2017-01-01'
end_date = '2017-01-31'
output_folder = '/project/foster/clouds/data/nasa/mod021km'

# Create list of dates

date_range = date_list(start_date,end_date)

for i, date_i in enumerate(sorted(date_range)):
    if i % size == rank:
        print('Launching date ', date_i.strftime("%Y-%m-%d"), rank, flush=True)
        download('MOD021KM', date_i.strftime("%Y-%m-%d"), date_i.strftime("%Y-%m-%d"), 90.0, -90.0, 180.0, -180.0, 6, output_folder, cpus=6)




