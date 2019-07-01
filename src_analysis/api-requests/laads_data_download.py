#!/usr/bin/env python

#
# Attempts to do HTTP Gets urllib.requets(py3) or subprocess
# if tlsv1.1+ isn't supported by the python ssl module
#
# Will download csv or json depending on which python module is available
#

from __future__ import (division, print_function, absolute_import, unicode_literals)

import os
import os.path
import shutil
import sys
import ssl

try:
    from StringIO import StringIO   # python2
except ImportError:
    from io import StringIO         # python3


################################################################################


USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n', '').replace('\r', '')


def geturl(url, token=None, out=None):
    headers = {'user-agent' : USERAGENT}
    if not token is None:
        headers['Authorization'] = 'Bearer ' + token
    try:
        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        from urllib.request import urlopen, Request
        fh = urlopen(Request(url, headers=headers), context=CTX)
        if out is None:
            return fh.read().decode('utf-8')
        else:
            shutil.copyfileobj(fh, out)
    except:
        geturl(url, token, out)


################################################################################

#This script will recursively download all files if they don't exist from a
#LAADS URLand stores them to the specified path


def sync(src, dest, tok):
    '''synchronize src url with dest directory'''
    try:
        import csv
        files = [f for f in csv.DictReader(StringIO(geturl('%s.csv' % src, tok)),
                                           skipinitialspace=True)]
    except ImportError:
        import json
        files = json.loads(geturl(src + '.json', tok))
    # use os.path since python 2/3 both support it while pathlib is 3.4+
    for f in files:
        # currently we use filesize of 0 to indicate directory
        filesize = int(f['size'])
        path = os.path.join(dest, f['name'])
        url = src + '/' + f['name']
        if filesize == 0:
            try:
                print('creating dir:', path)
                os.mkdir(path)
                sync(src + '/' + f['name'], path, tok)
            except IOError as e:
                print("mkdir `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
        else:
            try:
                if not os.path.exists(path):
                    print('downloading: ', path)
                    with open(path, 'w+b') as fh:
                        geturl(url, tok, fh)
                else:
                    print('skipping: ', path)
            except IOError as e:
                print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
