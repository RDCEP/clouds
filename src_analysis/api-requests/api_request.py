'''
Katy Koenig
June 2019

Functions to request downloads of modis data from NASA API
'''
import os
import requests
from bs4 import BeautifulSoup
import csv
import time
import pandas as pd
import laads_data_download as ldd


DATE_FILE = 'label1.txt'
COORDINATES_FILE = 'coords.csv'
EMAIL = 'koenig1@uchicago.edu'
APP_KEY = '126AA2A4-96BA-11E9-9D2C-D7883D88392C'


### First we need to start by clearing/releasing all available orders #####

def clear_all_orders(email_address=EMAIL):
    '''
    Releases all orders for given account on NASA website

    Inputs:
        email_address(str): email address for NASA account

    Outputs: None
    '''
    alive_lst = get_alive_orders(email_address)
    for order in alive_lst:
        release_order(order, email_address)


def get_alive_orders(email_address):
    '''
    Finds all orders that have not been released

    Inputs:
        email_address(str): email address of accoutn previously registered at NASA

    Outputs:
        alive_orders: list of integers of alive order id numbers
    '''
    order_ids = []
    alive_orders = []
    params = {'email': email_address}
    orders = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/getAllOrders?', params)
    soup = BeautifulSoup(orders.content, 'html5lib')
    counter = 0
    for id in soup.find_all('return'):
        order_ids.append(id.text)
    for id in order_ids:
        status = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/getOrderStatus?orderId=' + str(id))
        soup = BeautifulSoup(status.content, 'html5lib')
        if soup.find('return').text == 'Available':
            alive_orders.append(id)
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
    resp = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/releaseOrder?', params)
    soup = BeautifulSoup(resp.content, 'html5lib')


### Now, make coords csv to fill in (feel free to delete any coordinates that are have downloads completed) ####

def write_csv(outputfile='coords.csv'):
    '''
    Writes csv of requested locations

    Inputs:
        outputfile(str): name of outputfile to be saved

    Outputs: None (saved file)
    '''
    with open(outputfile, 'w') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        outputwriter.writerow(['north', 'south', 'east', 'west'])
        outputwriter.writerow([32.5, 12.4, 127.7, -155.4])
        #outputwriter.writerow([-34.3, -49.9, 40.2, 23.5])
        #outputwriter.writerow([-19.6, -44.9, 14.2, -5.6])
        #outputwriter.writerow([42, 23.6, -48.1, -74.7])
        #outputwriter.writerow([33.6, 12.4, -15.9, -37.5])
        #outputwriter.writerow([-4, 34.5, -107.6, -137.3])
        #outputwriter.writerow([-6.5, -31.8, -72.3, -102.3])
        #outputwriter.writerow([32.6, 3.4, -109.6, -135.9])
    csvfile.close()


### To actually download images: you need only call combining_fn() located at bottom of file

def find_files(dates=DATE_FILE, coords=COORDINATES_FILE, email_address=EMAIL):
    '''
    Calls NASA LWS API to order downloads of specified files

    Inputs:
        dates(str): txt filename with desired dates
        coords(str): csv filename with desired coordinates
        email_address(str): email address previously registered to NASA site

    Outputs:
        total_orders: list of order ids (ints)
    '''
    # Initialize params for request
    search_params = {'products': 'MOD35_L2', 'collection': 61, 'dayNightBoth': 'DB', 'coordsOrTiles': 'coords'}
    # Add additional params of locations
    dates_file = open(dates, 'r')
    label_dates = dates_file.read().split('\n')
    coords_df = pd.read_csv(coords)
    order_ids = []
    total_params =[]
    for row in coords_df.iterrows():
        search_params['north'] = row[1][0]
        search_params['south'] = row[1][1]
        search_params['east'] = row[1][2]
        search_params['west'] = row[1][3]
        # Iterate through date list of each location
        for date in label_dates[:-1]:
            search_params['startTime'] = str(date) + ' 00:00:00'
            search_params['endTime'] = str(date) + ' 23:59:59'
            # Find relevant files
            response = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/searchForFiles?', search_params)
            soup = BeautifulSoup(response.content, 'html5lib')
            file_ids = []
            for id in soup.find_all('return'):
                file_ids.append(id.text)
            # Order downloads of files
            order_params = {'email': email_address, 'fileIds': ','.join(file_ids)}
            total_params.append(order_params)
    return total_params


def batch_order_and_delete(total_params):
    '''
    Orders and downloads batches for files due to the NASA request limit

    Inputs:
        total_params: list of dictionaries of parameters to be passed

    Outputs: list of order ids that were ordered, downloaded and released
    '''
    order_ids = []
    max_size = 100
    for i in range(0, len(total_params), max_size):
        print(i)
        chunk = total_params[i:i+max_size]
        for order_param in chunk:
            order_files(order_param, order_ids)
        print('Waiting for Availability to Download')
        download_order(order_ids)
        print('Releasing orders')
        clear_all_orders()
    return order_ids


def order_files(order_params, order_ids):
    '''
    Places orders via NASA API to for downloading data and updates order_ids list to reflect new orders

    Inputs:
        order_params: dictionary of parameters to pass to build url
        order_ids: list of integers

    Outputs: None (modified order_ids list in place)
    '''
    order_response = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/orderFiles?', order_params)
    if order_response.status_code == 200:
        order_soup = BeautifulSoup(order_response.content, 'html5lib')
        order_ids.append(int(order_soup.find('return').text))
    else:
        print(order_response.status_code)
        time.sleep(1)
        order_files(order_params, order_ids)


def download_order(order_lst, destination='hdf_files', token=APP_KEY, email_address=EMAIL):
    '''
    Checks to see if order status complete; when complete, downloads files in order

    Inputs:
        order_lst: list of integers, each representing an order placed
        destination(str): folder name in which to save images
        token(str): app key from NASA

    Outputs: None (saved images)
    '''
    if not os.path.exists(destination):
        os.makedirs(destination)
    for order in order_lst:
        response = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/getOrderStatus?orderId=' + str(order))
        soup = BeautifulSoup(response.content, 'html5lib')
        status = soup.find('return')
        # Continuously checks if order processed & downloads if ready
        if status.text == 'Available':
            source = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/' + str(order) + '/'
            response = requests.get(source)
            if response.status_code == 200:
                ldd.sync(source, destination, token)
            else:
                order_lst.append(order)
        else:
            order_lst.append(order)


def combining_fn():
    '''
    Combining function to search, order, download and release all files in batches

    Inputs:

    Outputs:
    '''
    order_params = find_files(dates=DATE_FILE, coords=COORDINATES_FILE, email_address=EMAIL)
    order_ids = batch_order_and_delete(order_params)
    return order_ids
