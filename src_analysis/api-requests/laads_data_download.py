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
import time
import requests

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

    from urllib.request import urlopen, Request, URLError, HTTPError
    fh = urlopen(Request(url, headers=headers), context=CTX)
    print(fh)
    if out is None:
        return fh.read().decode('utf-8')
    else:
        shutil.copyfileobj(fh, out)
    # except:
    #     time.sleep(1)
    #     response = requests.get(url)
    #     print(response.status_code)
    #     geturl(url, token, out)
            # except HTTPError as e:
            #     print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
            #     print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            # except URLError as e:
            #     print('Failed to make request: %s' % e.reason, file=sys.stderr)

    # except AttributeError:
    #     # OS X Python 2 and 3 don't support tlsv1.1+ therefore... curl
    #     import subprocess
    #     try:
    #         args = ['curl', '--fail', '-sS', '-L', '--get', url]
    #         for (k,v) in headers.items():
    #             args.extend(['-H', ': '.join([k, v])])
    #         if out is None:
    #             # python3's subprocess.check_output returns stdout as a byte string
    #             result = subprocess.check_output(args)
    #             return result.decode('utf-8') if isinstance(result, bytes) else result
    #         else:
    #             subprocess.call(args, stdout=out)
    #     except subprocess.CalledProcessError as e:
    #         print('curl GET error message: %' + (e.message if hasattr(e, 'message') else e.output), file=sys.stderr)



################################################################################


DESC = "This script will recursively download all files if they don't exist from a LAADS URL and stores them to the specified path"


def sync(src, dest, tok):
    '''synchronize src url with dest directory'''
    print('nsync')
    try:
        import csv
        files = [ f for f in csv.DictReader(StringIO(geturl('%s.csv' % src, tok)), skipinitialspace=True) ]
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
                    print('downloading: ' , path)
                    with open(path, 'w+b') as fh:
                        geturl(url, tok, fh)
                else:
                    print('skipping: ', path)
            except IOError as e:
                print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)


def _main(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description=DESC)
    parser.add_argument('-s', '--source', dest='source', metavar='URL', help='Recursively download files at URL', required=True)
    parser.add_argument('-d', '--destination', dest='destination', metavar='DIR', help='Store directory structure in DIR', required=True)
    parser.add_argument('-t', '--token', dest='token', metavar='TOK', help='Use app token TOK to authenticate', required=True)
    args = parser.parse_args(argv[1:])
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    return sync(args.source, args.destination, args.token)


if __name__ == '__main__':
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
