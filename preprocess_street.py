import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import zipfile
from random import sample
from shutil import copy2
import shutil
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dhs_labels import get_dhs_labels

import imghdr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dhs_labels_df = get_dhs_labels()
dhs_ids_valid_water_index = dhs_labels_df.index[dhs_labels_df["water_index"].notna()]

unzip_dir = "data/street_imgs_raw/unzipped"
save_path = "data/street_npzs_notna"

# dataset split
cnames = ["MD", "BD", "CD", "CM", "GH", "ZW", "NP", "TJ", "BJ", "BO", "AM", "AO"]
train = ["MD", "BD", "CD", "CM", "GH", "ZW", "NP", "TJ"]
val = ["BJ", "BO"]
test = ["AM", "AO"]

# params for processing images
model_image_size = (256, 256)

def preprocess_image(img_path, model_image_size):
  '''
  adapted from coursera Deep Learning course, CNN module. Programming assignment: Car detection
  https://www.coursera.org/learn/convolutional-neural-networks/ungradedLab/PbGsA/car-detection-with-yolo/lab?path=%2Fedit%2Fweek3%2FCar%2520detection%2520for%2520Autonomous%2520Driving%2Fyolo_utils.py

  img_path: relative or absolute path to the image. 
    For example, /content/drive/My Drive/ColabNotebooks/street_imgs/AL/AL-2017-7#-00000127/1184779192036638.jpeg
  model_image_size: a tuple indicating the desired dimensions for the output image data. 
    For example, model_image_size = (608, 608). Then output shape is (608, 608, 3)
  '''
  image_type = imghdr.what(img_path)
  image = Image.open(img_path)
  resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
  image_data = np.array(resized_image, dtype='float32')
  image_data /= 255.
  image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
  return image_data

def process_imgs_per_satellite_group(street_imgs_path, save_path, model_image_size, cname_dir_lock):
  '''
  For example,
    street_imgs_path = /content/drive/My Drive/ColabNotebooks/street_imgs/AL/AL-2017-7#-00000127
  '''
  dhsid = street_imgs_path.split("/")[-1]
  cname = street_imgs_path.split("/")[-2]

  # print(f"processing images for {street_imgs_path}...")
  for jpeg_img_name in os.listdir(street_imgs_path):
    if not jpeg_img_name.endswith(".jpeg"): 
      continue

    jpeg_img_path = os.path.join(street_imgs_path, jpeg_img_name)
    img_id = jpeg_img_name.split(".")[0]
    save_npz_name = f"{dhsid}-{img_id}"
    cname_save_path = os.path.join(save_path, cname)
    save_npz_path = os.path.join(cname_save_path, save_npz_name)
    npz_file_path = f"{save_npz_path}.npz"
    if (os.path.exists(npz_file_path)):
      continue
    
    img_data = preprocess_image(img_path=jpeg_img_path, model_image_size=model_image_size)
    
    # save img_data as a npz file
    cname_dir_lock.acquire()
    if (not os.path.isdir(cname_save_path)):
      os.mkdir(cname_save_path)
    cname_dir_lock.release()
    np.savez_compressed(save_npz_path, img_data=img_data) 

    # os.remove(jpeg_img_path) # delete this image once its data is saved
  
  return f"Done preprocessing and saving data for {dhsid}"
  
def process_imgs_per_country(country_unzipped_directory, model_image_size):
  '''
  all paths are absolute
  For example, 
    country_unzipped_directory = /content/drive/My Drive/ColabNotebooks/street_imgs/AL
    model_image_size = (608, 608)

  Only process street images whose their satellite image has valid water quality index 
  '''
  country_name = country_unzipped_directory.split("/")[-1] #e.g. AL

  '''
  Each country has a folder for street-level images. Images are in JPEG format.
  They're grouped by the dhs ID of the corresponding satellite image.
  '''

  print(f"Preprocessing and saving street image data for {country_name}...")
  for satellite_img_dhs_id in tqdm(os.listdir(country_unzipped_directory)):
    street_imgs_path = os.path.join(country_unzipped_directory, satellite_img_dhs_id)

    # filter out satellite images that have missing water quality index
    if satellite_img_dhs_id in dhs_ids_valid_water_index:
      process_imgs_per_satellite_group(street_imgs_path, save_path, model_image_size)
    else: # delete the folder for this satellite group
      try:
        print(f"Removing {street_imgs_path}")
        shutil.rmtree(street_imgs_path)
      except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

  print(f"Done preprocessing and saving street image data for {country_name}")
    
def process_imgs_per_country_parallel(country_unzipped_directory, model_image_size, cname_dir_lock):
    '''
    all paths are absolute
    For example, 
    country_unzipped_directory = /content/drive/My Drive/ColabNotebooks/street_imgs/AL
    model_image_size = (608, 608)

    Only process street images whose their satellite image has valid water quality index 
    '''
    country_name = country_unzipped_directory.split("/")[-1] #e.g. AL
    print(f"Preprocessing images for {country_name}")

    '''
    Each country has a folder for street-level images. Images are in JPEG format.
    They're grouped by the dhs ID of the corresponding satellite image.
    '''
  
    street_imgs_path_list = []
    for satellite_img_dhs_id in os.listdir(country_unzipped_directory):
        street_imgs_path = os.path.join(country_unzipped_directory, satellite_img_dhs_id)
        # filter out satellite images that have missing water quality index
        if satellite_img_dhs_id in dhs_ids_valid_water_index:
            street_imgs_path_list.append(street_imgs_path)
        else: # delete the folder for this satellite group
            try:
                print(f"Removing {street_imgs_path}")
                shutil.rmtree(street_imgs_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
  
    with ThreadPoolExecutor(max_workers=8) as sub_executor:
        sub_futures = []
        for street_imgs_path in street_imgs_path_list:
            sub_futures.append(sub_executor.submit(process_imgs_per_satellite_group, 
                                    street_imgs_path=street_imgs_path, 
                                    save_path=save_path, 
                                    model_image_size=model_image_size,
                                    cname_dir_lock=cname_dir_lock))
        for sub_future in as_completed(sub_futures):
            print(sub_future.result())

    return f"Done preprocessing and saving street image data for {country_name}"
    
if __name__ == "__main__":
    street_dir_to_preprocess_list = []
    cname_dir_locks = {}

    # for cname in cnames:
    for cname in ["AL"]:
      country_unzipped_directory = os.path.join(unzip_dir, cname)
      street_dir_to_preprocess_list.append(country_unzipped_directory)
      cname_dir_locks[cname] = threading.Lock()

    with ThreadPoolExecutor(max_workers=12) as preprocess_executor:
      preprocess_futures = []
      for country_unzipped_directory in street_dir_to_preprocess_list:
        cname = country_unzipped_directory.split("/")[-1]
        preprocess_futures.append(preprocess_executor.submit(process_imgs_per_country_parallel, 
                                            country_unzipped_directory=country_unzipped_directory, 
                                            model_image_size=model_image_size, 
                                            cname_dir_lock=cname_dir_locks[cname]))
      for preprocess_future in as_completed(preprocess_futures):
        print(preprocess_future.result())


