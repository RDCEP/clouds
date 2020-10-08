#
import os
import re
import glob
import shutil


def read_txt(filename):
  with open(filename, 'r') as f:
    lines = f.read().split("\n")
    data = []
    for line in lines:
      if len(line) > 0:
        data.append(line)

  return data


def ripper(data,bname='A2008'):
  """ A2008.001.0720.nc """
  # ref: https://python.keicode.com/lang/regular-expression-findall.php
  numerics = []
  for i in data:
    n = re.findall(r'[0-9]{3}.[0-9]{4}', i)
    numerics.append(bname+n[0])

  return numerics


def run_sys(filename=None,hdf_datadir=None,savedir=None, verbose=False,bname=False):

  data_list = read_txt(filename)
  print(f"Check Number of Files {len(data_list)}", flush=True)

  ripped_name = ripper(data_list,bname)
  if verbose:
    for i in ripped_name:
      print(i)

  # make dir
  os.makedirs(savedir, exist_ok=True)

  # operation
  filelist =glob.glob(os.path.join(hdf_datadir, "*.hdf"))
  for ifile in filelist:
    #print(os.path.basename(ifile),flush=True)
    for iname in ripped_name:
      if iname in os.path.basename(ifile):
        print(os.path.basename(ifile), flush=True)
        
        shutil.move(ifile, savedir)
        

  print("NORMAL END", flush=True)


if __name__ == "__main__":
  # PARAM
  product = "MYD02"
  product2 = "MYD021KM"
  bname    = "A2008" 
  
  hdf_datadir = "/home/tkurihana/Research/data/"+product+"/"+product2+"/2008/001"

  # MAIN
  run_sys(filename='/home/tkurihana/Research/data3/200801/day/nclist.txt',
          hdf_datadir=hdf_datadir,
          savedir="/home/tkurihana/Research/data/"+product+"/20080101",
          bname=bname
  )

