import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import zipfile
from random import sample
from shutil import copy2
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# get DHS labels
def get_dhs_labels():
    dhs_labels_df = pd.read_csv("data/dhs_final_labels.csv")
    dhs_labels_df['survey'] = dhs_labels_df['DHSID_EA'].str[:10]
    dhs_labels_df['cc'] = dhs_labels_df['DHSID_EA'].str[:2]
    dhs_labels_df['path'] = 'dhs_npzs/' + dhs_labels_df['survey'] + '/' + dhs_labels_df['DHSID_EA'] + '.npz'
    dhs_labels_df.set_index('DHSID_EA', verify_integrity=True, inplace=True)

    return dhs_labels_df

'''
========= Unzip street image files ===========
'''
street_path_source = "data/street_imgs_raw"
unzip_dir = "data/street_imgs_raw"

dhs_labels_df = get_dhs_labels()
dhs_ids_valid_water_index = dhs_labels_df.index[dhs_labels_df["water_index"].notna()]

street_dir_to_unzip_list = [] # path to the zipfile by country. For example, /content/drive/My Drive/ColabNotebooks/dhs/mapillary/AL.zip
to_unzip_cnames = ["MD", "BD", "CD", "CM", "GH", "ZW", "NP", "TJ", "BJ", "BO", "AM", "AO"]
to_unzip_cnames = ["AL"]

def unzip_file(zfile_path, directory_to_extract_to):
  '''
  given a path to a zipfile, unzip it and put in `directory_to_extract_to` 
  takes in absolute paths
  For example, 
    zfile = "/content/drive/My Drive/ColabNotebooks/dhs/mapillary/AL.zip"
    directory_to_extract_to = "/content/drive/My Drive/ColabNotebooks/street_imgs/"
  '''
  zipped_fname = zfile_path.split("/")[-1]
  country_name = zipped_fname.split(".")[0]

  print(f"Extracting {zipped_fname} ...")
  with zipfile.ZipFile(zfile_path, 'r') as zip_ref:
    for file_name in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())): 
      zip_ref.extract(file_name, path=directory_to_extract_to)
  
  print(f"Done extracting {zipped_fname}")

def unzip_file_parallel(zfile_path, directory_to_extract_to):
  '''
  given a path to a zipfile, unzip it and put in `directory_to_extract_to` 
  takes in absolute paths
  For example, 
    zfile = "/content/drive/My Drive/ColabNotebooks/dhs/mapillary/AL.zip"
    directory_to_extract_to = "/content/drive/My Drive/ColabNotebooks/street_imgs_parallel/"
  '''
  zipped_fname = zfile_path.split("/")[-1]
  country_name = zipped_fname.split(".")[0]

  print(f"Extracting {zipped_fname} ...")
  file_list = []
  with zipfile.ZipFile(zfile_path, 'r') as zip_ref:
    for file_name in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())): 
      file_list.append(file_name)
  
  with zipfile.ZipFile(zfile_path, 'r') as zip_ref:
    with ThreadPoolExecutor(max_workers=20) as unzip_sub_executor:
      unzip_sub_futures = []
      for file_name in file_list:
        unzip_sub_futures.append(unzip_sub_executor.submit(zip_ref.extract, member=file_name, path=directory_to_extract_to))
      for unzip_sub_future in as_completed(unzip_sub_futures):
        print(unzip_sub_future.result())
 
  return f"Done extracting {zipped_fname}"


if __name__ == "__main__":
  street_dir_to_unzip_list = []
  for fname in os.listdir(street_path_source):
    if not fname.endswith(".zip"): 
      continue
    country_name = fname.split(".")[0]
    if country_name in to_unzip_cnames:
      street_zfile_path = os.path.join(street_path_source, fname)
      street_dir_to_unzip_list.append(street_zfile_path)
  
  with ThreadPoolExecutor(max_workers=12) as unzip_executor:
    unzip_futures = []
    for street_zfile_path in street_dir_to_unzip_list:
      unzip_futures.append(unzip_executor.submit(unzip_file, zfile_path=street_zfile_path, directory_to_extract_to=unzip_dir))
    for unzip_future in as_completed(unzip_futures):
      print(unzip_future.result())