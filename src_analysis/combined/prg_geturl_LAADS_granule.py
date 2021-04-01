#!/usr/bin/env python
 
# script supports either python2 or python3
#
# Attempts to do HTTP Gets with urllib2(py2) urllib.requets(py3) or subprocess
# if tlsv1.1+ isn't supported by the python ssl module
#
# Will download csv or json depending on which python module is available
#
 
from __future__ import (division, print_function, absolute_import, unicode_literals)
 
import argparse
import os
import os.path
import shutil
import sys
import pickle
 
try:
    from StringIO import StringIO   # python2
except ImportError:
    from io import StringIO         # python3
 
 
################################################################################
 
 
USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n','').replace('\r','')
 
 
def geturl(url, token=None, out=None):
    headers = { 'user-agent' : USERAGENT }
    if not token is None:
        headers['Authorization'] = 'Bearer ' + token
    try:
        import ssl
        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        if sys.version_info.major == 2:
            import urllib2
            try:
                fh = urllib2.urlopen(urllib2.Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read()
                else:
                    shutil.copyfileobj(fh, out)
            except urllib2.HTTPError as e:
                print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            except urllib2.URLError as e:
                print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None
 
        else:
            from urllib.request import urlopen, Request, URLError, HTTPError
            while True:
                try:
                    fh = urlopen(Request(url, headers=headers), context=CTX, timeout=600)
                    if not isinstance(fh, int):
                        if fh.getcode() == 200:
                            if out is None:
                                return fh.read().decode('utf-8')
                            else:
                                shutil.copyfileobj(fh, out)
                            break
                except HTTPError as e:
                    print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                    print('HTTP GET error message: %s' % e.message, file=sys.stderr)
                except URLError as e:
                    print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None
 
    except AttributeError:
        # OS X Python 2 and 3 don't support tlsv1.1+ therefore... curl
        import subprocess
        try:
            args = ['curl', '--fail', '-sS', '-L', '--get', url]
            for (k,v) in headers.items():
                args.extend(['-H', ': '.join([k, v])])
            if out is None:
                # python3's subprocess.check_output returns stdout as a byte string
                result = subprocess.check_output(args)
                return result.decode('utf-8') if isinstance(result, bytes) else result
            else:
                subprocess.call(args, stdout=out)
        except subprocess.CalledProcessError as e:
            print('curl GET error message: %' + (e.message if hasattr(e, 'message') else e.output), file=sys.stderr)
        return None
 
################################################################################

def preload(src,tok,thres=100):
    '''synchronize src corresponding with modis terra/aqua 021KM
       output: return indices of MOD02 >= 100
    '''
    try:
        import csv
        files = [ f for f in csv.DictReader(StringIO(geturl('%s.csv' % src, tok)), skipinitialspace=True) ]
    except ImportError:
        import json
        files = json.loads(geturl(src + '.json', tok))
    
    return files 


################################################################################
 

 
DESC = "This script will recursively download all files if they don't exist from a LAADS URL and stores them to the specified path"
 
 
def sync(src, dest, tok, m2src, granules, product):
    '''synchronize src url with dest directory'''
    # filename setup 
    filename = granules[0]
    timestamp= granules[1].split(',')
    ssrc = os.path.join(src, str(timestamp[0]), str(timestamp[1]) ) + '/'  # https://www.sejuku.net/blog/64408

    # save directory setup
    ddest = os.path.join(dest, product,  str(timestamp[0]), str(timestamp[1]))
    os.makedirs(ddest, exist_ok=True)

    try:
        import csv
        files = [ f for f in csv.DictReader(StringIO(geturl('%s.csv' % ssrc, tok)), skipinitialspace=True) ]
    except ImportError:
        import json
        files = json.loads(geturl(src + '.json', tok))
    
    
    # select file fit in granules
    sm2src = os.path.join(m2src, str(timestamp[0]), str(timestamp[1]) )  + '/'
    mod02s = preload(sm2src,tok)
    sfiles = [ files[i] for i, mod02 in enumerate(mod02s) if mod02['name'] == filename ]
 
    # use os.path since python 2/3 both support it while pathlib is 3.4+
    for f in sfiles:
        # currently we use filesize of 0 to indicate directory
        filesize = int(f['size'])
        path = os.path.join(ddest, f['name'])
        url = ssrc + '/' + f['name']
        if filesize == 0:
            try:
                print('creating dir:', path, flush=True)
                os.mkdir(path)
                sync(ssrc + '/' + f['name'], path, tok)
            except IOError as e:
                print("mkdir `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
        else:
            try:
                if not os.path.exists(path):
                    print('downloading: ' , path, flush=True)
                    with open(path, 'w+b') as fh:
                        geturl(url, tok, fh)
                else:
                    print('skipping: ', path, flush=True)
            except IOError as e:
                print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
    return 0
 
 
def _main(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description=DESC)
    parser.add_argument('-s', '--source', dest='source', metavar='URL', help='Recursively download files at URL', required=True)
    parser.add_argument('-ms','--m2source', dest='m2source', metavar='URL', help='Recursively download corresponding MOD02 files at URL', required=True)
    parser.add_argument('-d', '--destination', dest='destination', metavar='DIR', help='Store directory structure in DIR', required=True)
    parser.add_argument('-t', '--token', dest='token', metavar='TOK', help='Use app token TOK to authenticate', required=True)
    parser.add_argument('-g', '--granule', dest='granule', metavar='GRN', help='Load pickle file to download granule scale', required=True)
    parser.add_argument('-p', '--product', dest='product', metavar='PRD', help='MODIS product name', required=True)
    args = parser.parse_args(argv[1:])
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    with open(args.granule,'rb') as f:
        granules = pickle.load(f) # load dict
    
    for granule, dates in granules.items():
        r = sync(args.source, args.destination, args.token, args.m2source, (granule,dates), args.product)
    return r
 
if __name__ == '__main__':
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
