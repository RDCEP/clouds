'''
Katy Koenig
June 2019

Functions to request downloads of modis data from NASA API
'''
import os
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import laads_data_download as ldd


DATE_FILE = 'label1.txt'
COORDINATES_FILE = 'coords.csv'
WSDL_FILE = 'https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices?wsdl'
ACCESS_POINT = 'http://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices'
EMAIL = 'koenig1@uchicago.edu'
APP_KEY = '126AA2A4-96BA-11E9-9D2C-D7883D88392C'

def request_downloads(dates=DATE_FILE, coords=COORDINATES_FILE, url=WSDL_FILE, email_address=EMAIL):
    '''
    Calls NASA LWS API to order downloads of specified files

    Inputs:
        dates(str): txt filename with desired dates
        coords(str): csv filename with desired coordinates
        url(str): website for public interface to be queried
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
    total_orders = []
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
            for id in soup.find_all('return'):
                total_orders.append(int(id.text))
            return total_orders
            
            # Order downloads of files

    #         order_ids = requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/orderFiles?', order_params)
    #         total_orders += order_ids
    # return total_orders


#https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/orderFiles?orderIds=[2760357367]&email=koenig1@uchicago.edu



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
        if status.text == 'Available':
            source = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/' + str(order) + '/'
            ldd.sync(source, destination, token)
            #params = {'order': order, 'email': email_address}
            # To delete -- NASA not working
            #requests.get('https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/releaseOrder?', params)
        else:
            order_lst.append(order)


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
        outputwriter.writerow([-34.3, -49.9, 40.2, 23.5])
        outputwriter.writerow([-19.6, -44.9, 14.2, -5.6])
        outputwriter.writerow([42, 23.6, -48.1, -74.7])
        outputwriter.writerow([33.6, 12.4, -15.9, -37.5])
        outputwriter.writerow([-4, 34.5, -107.6, -137.3])
        outputwriter.writerow([-6.5, -31.8, -72.3, -102.3])
        outputwriter.writerow([32.6, 3.4, -109.6, -135.9])
    csvfile.close()
