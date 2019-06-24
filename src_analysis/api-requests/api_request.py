'''
Katy Koenig
June 2019

Functions to request downloads of modis data from NASA API
'''
from zeep import Client, xsd, Settings
import csv
import pandas as pd


DATE_FILE = 'label1.txt'
COORDINATES_FILE = 'coords.csv'
WSDL_FILE = 'https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices?wsdl'
ACCESS_POINT = 'http://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices'
EMAIL = 'koenig1@uchicago.edu'

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
    search_params = {'products': 'MOD35_L2', 'collection': 61, 'dayNightBoth': 'DB'}
    order_params = {'doMosaic': True}
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
            # API Request for each set of parameters
            # Initialize interface for SOAP interactions
            sets = Settings(raw_response=True, xml_huge_tree=True)
            client = Client(wsdl, settings=sets)
            # Find relevant files
            file_id_lst = client.service.searchForFiles(**p)
            # Order downloads of files
            order_ids = service.OrderFiles(email=email_address, fileIds=file_id_lst, doMosaic=True)
            total_orders += order
    return total_orders
    

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
