'''
Katy Koenig
June 2019

Functions to request downloads of modis data from NASA API
'''

import os
from sys import argv
import time
import datetime
import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
import laads_data_download as ldd


########################################################
#INPUT REGISTERED NASA EMAIL AND APP KEY IN QUOTES BELOW
EMAIL = 'koenig1@uchicago.edu'
APP_KEY = '126AA2A4-96BA-11E9-9D2C-D7883D88392C'
########################################################

### MAKE SURE YOU HAVE THE RIGHT FILENAMES FOR COORDINATES_FILE AND DATE_FILE
DATE_FILE = ''
COORDINATES_FILE = ''

# WHERE DO YOU WANT THE FILES TO BE SAVED?
DESIRED_DIR = None

### Function to make csv of coordinates for patches ####
### Feel free to add/delete any coordinates as needed ####

def write_csv(outputfile='coords.csv'):
    '''
    Writes csv of requested locations

    Inputs:
        outputfile(str): name of outputfile to be saved

    Outputs: None (saved file)
    '''
    with open(outputfile, 'w') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        outputwriter.writerow(['name', 'north', 'south', 'east', 'west'])
        outputwriter.writerow(['open_pacific', 32.5, 12.4, 127.7, -155.4])
        outputwriter.writerow(['open_south_sf', -34.3, -49.9, 40.2, 23.5])
        outputwriter.writerow(['closed_west_sf', -19.6, -44.9, 14.2, -5.6])
        outputwriter.writerow(['open_west_atlantic', 42, 23.6, -48.1, -74.7])
        outputwriter.writerow(['closed_east_atlantic', 33.6, 12.4, -15.9, -37.5])
        outputwriter.writerow(['open_chile', -4, -34.5, -107.6, -137.3])
        outputwriter.writerow(['closed_chile', -6.5, -31.8, -72.3, -102.3])
        outputwriter.writerow(['closed_california', 32.6, 3.4, -109.6, -135.9])
    csvfile.close()


### Functions to clear/release previous orders #####

def clear_all_orders(email_address=EMAIL):
    '''
    Releases all orders for given account on NASA website

    Inputs:
        email_address(str): email address for NASA account

    Outputs: None
    '''
    alive_lst = get_alive_orders(email_address)
    if alive_lst:
        print('Releasing Orders:')
        for order in alive_lst:
            print(order)
            release_order(order, email_address)


def get_alive_orders(email_address):
    '''
    Finds all orders that have not been released

    Inputs:
        email_address(str): email address of account previously registered at NASA

    Outputs:
        alive_orders: list of integers of alive order id numbers
    '''
    order_ids = []
    alive_orders = []
    params = {'email': email_address}
    orders = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/getAllOrders?',
                          params)
    soup = BeautifulSoup(orders.content, 'html5lib')
    counter = 0
    for o_id in soup.find_all('return'):
        order_ids.append(o_id.text)
    for o_id in order_ids:
        status = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/getOrderStatus?orderId=' + str(o_id))
        soup = BeautifulSoup(status.content, 'html5lib')
        if soup.find('return').text != 'Canceled':
            alive_orders.append(o_id)
        else:
            counter += 1
            if counter == 10:
                return alive_orders


def release_order(order, email_address=EMAIL):
    '''
    Releases data order; Note that it may take minutes for NASA webpage to be updated

    Inputs:
        order(int): id number of order
        email_address(str): NASA-registered email address associated with order

    Outputs: None
    '''
    params = {'orderId': order, 'email': email_address}
    requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/releaseOrder?', params)


### To actually download images: you need only call combining_fn() located at bottom of file

def find_files(prods='MOD06_L2--61', dates=DATE_FILE, coords=COORDINATES_FILE, email_address=EMAIL):
    '''
    Calls NASA LWS API to order downloads of specified files

    Inputs:
        dates(str): txt filename with desired dates
        coords(str): csv filename with desired coordinates
        email_address(str): email address previously registered to NASA site

    Outputs:
        total_params: list of dictionaries of parameters
    '''
    # Initialize params for request
    bad_dates = pd.read_csv('total_bad_dates.csv')
    destination_lst = []
    search_params = {'products': prods,
                     'collection': 61,
                     'dayNightBoth': 'DB',
                     'coordsOrTiles': 'coords'}
    # Add additional params of locations
    dates_file = open(dates, 'r')
    label_dates = dates_file.read().split('\n')
    coords_df = pd.read_csv(coords)
    total_params = []
    for row in coords_df.iterrows():
        location = row[1][0]
        search_params['north'] = row[1][1]
        search_params['south'] = row[1][2]
        search_params['east'] = row[1][3]
        search_params['west'] = row[1][4]
        bad_start = datetime.datetime.strptime('08-29-2006', '%m-%d-%Y')
        bad_end = datetime.datetime.strptime('08-04-2008', '%m-%d-%Y')
        # Iterate through date list of each location
        for date in label_dates[:-1]:
            if date not in bad_dates['date']:
                # Checks to ensure that in bad time period (when issue w/ detector)
                dt_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
                if not bad_start < dt_obj < bad_end:
                    search_params['startTime'] = date + ' 00:00:00'
                    search_params['endTime'] = date + ' 23:59:59'
                    # Find relevant files
                    response = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/searchForFiles?',
                                            search_params)
                    soup = BeautifulSoup(response.content, 'html5lib')
                    file_ids = []
                    for f_id in soup.find_all('return'):
                        order_txt = f_id.text
                        # Checks to ensure valid order number
                        if len(order_txt) == 10:
                            if order_txt != 'No results':
                                file_ids.append(order_txt)
                    # Order downloads of files
                    if file_ids:
                        order_params = {'email': email_address, 'doMosaic': 'True', 'fileIds': ','.join(file_ids)}
                        total_params.append(order_params)
                        destination = DESIRED_DIR + str(prods) + '/clustering' + str(location)
                        #destination = 'data/' + str(prods) + '/clustering/' + str(location) + '/' + str(date)
                        destination_lst.append(destination)
    return total_params, destination_lst


def batch_order_and_delete(total_params, destination_lst, token=APP_KEY, email_address=EMAIL):
    '''
    Orders and downloads batches for files due to the NASA request limit

    Inputs:
        total_params: list of dictionaries of parameters to be passed
        destination_lst: list of strings of foldernames

    Outputs: list of order ids that were ordered, downloaded and released
    '''
    order_ids = []
    max_size = 100
    for i in range(0, len(total_params), max_size):
        chunk = total_params[i:i+max_size]
        dest_chunk = destination_lst[i:i+max_size]
        print('Requesting Orders')
        for order_param in chunk:
            order_files(order_param, order_ids)
        print('Waiting for Availability to Download')
        # NASA takes minutes to prepare files
        time.sleep(300)
        for i in range(len(order_ids)):
            order = order_ids[i]
            destination = dest_chunk[i]
            download_order(order, destination, token)
        clear_all_orders(email_address)
    return order_ids


def order_files(order_params, order_ids):
    '''
    Places orders via NASA API to for downloading data and updates order_ids
        list to reflect new orders

    Inputs:
        order_params: dictionary of parameters to pass to build url
        order_ids: list of integers

    Outputs: None (modified order_ids list in place)
    '''
    order_response = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/orderFiles?',
                                  order_params)
    if order_response.status_code == 200:
        order_soup = BeautifulSoup(order_response.content, 'html5lib')
        order_ids.append(int(order_soup.find('return').text))
    else:
        print(order_response.text)
        print(order_params)
        print("Retrying NASA orderFiles API")
        time.sleep(5)
        order_files(order_params, order_ids)


def download_order(order, destination, token=APP_KEY):
    '''
    Downloads previously ordered file (tries continuously until file available)

    Inputs:
        order(int): order number
        destination(str): lfolder name to save file
        token(str): app key from NASA

    Outputs: None (saved images)
    '''
    try:
        source = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/' + str(order) + '/'
        if not os.path.exists(destination):
            os.makedirs(destination)
        ldd.sync(source, destination, token)
    except:
        time.sleep(3)
        print('trying again to download')
        download_order(order, destination, token)


def combining_fn(email_address=EMAIL, token=APP_KEY, dates=DATE_FILE,
                 coords=COORDINATES_FILE, products='MOD06_L2'):
    '''
    Combining function to search, order, download and release all files in batches

    Inputs:
        dates(str): txt filename with desired dates
        coords(str): csv filename with desired coordinates
        email_address(str): email address previously registered to NASA site

    Outputs: set of order_ids
    '''
    # First, releases all orders to make space for new orders
    clear_all_orders(email_address)
    # Generates list of order parameters
    order_params, destination_lst = find_files(products, dates, coords, email_address)
    # Orders, downloads and releases files
    order_ids = batch_order_and_delete(order_params, destination_lst, token, email_address)
    return set(order_ids)

if __name__ == '__main__':
    email = argv[1]
    app_key = argv[2]
    prods = argv[3]
    order_ids = combining_fn(email_address=email, token=app_key, products=prods)
    print(order_ids)
