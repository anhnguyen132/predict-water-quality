import os
from os import listdir
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
from tensorflow import keras 
from tqdm.auto import tqdm
import scipy.stats
from tensorflow.keras.models import Model
from utils import format_input_img
from random import sample
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import functools

def extract_features_vector(data_path, extractor, key):
    x = np.load(data_path)[key] 
    if key == 'x': # this is satellite image data
        x = x.reshape((1,) + x.shape)
        x = format_input_img(x)

    return extractor.predict(x)

def concat_and_savez_features(sat_fpath, extractors, sat_street_map, concat_save_path, lock):
    # print(f"Extracting features for satellite img: {sat_fpath}")
    sat_feat = extract_features_vector(sat_fpath, extractors['sat'], key='x')

    street_feat_sum = np.zeros([1, 64])
    count = 0
    for street_fpath in sat_street_map[sat_fpath]:
        count +=1
        try:
            street_feat_sum += extract_features_vector(street_fpath, extractors['street'], key='img_data')
        except:
            count -= 1
            continue
    print(f"count for {sat_fpath}: {count}")
    street_feat_avg = street_feat_sum / len(sat_street_map[sat_fpath])
    concat_feat = np.concatenate((sat_feat, street_feat_avg), axis=1).reshape(128, )
    
    # For example, sat_fpath = "data/dhs_subset_npzs_has_street/train/CD/CD-2013-6#-00000007.npz"
    # sat_fpath.rsplit("/", 3)[1:] = ['train', 'CD', 'CD-2013-6#-00000007.npz']
    concat_feat_save_path = os.path.join(concat_save_path, *sat_fpath.rsplit("/", 3)[1:])
    country_path = concat_feat_save_path.rsplit("/", 1)[0]
    lock.acquire()
    if not os.path.isdir(country_path):
        os.mkdir(country_path)
    lock.release()

    np.savez_compressed(concat_feat_save_path, x=concat_feat)
    return f"Done aggregating and saving features for {sat_fpath}"

def concat_and_savez_features_wrapper(dhs_id, sat_file_paths, lock, sample_size, extractors, sat_street_map, data_path):
    if len(sat_file_paths) > sample_size:
        sampled_sat_file_paths = sample(sat_file_paths, sample_size)
    else:
        sampled_sat_file_paths = sat_file_paths
    
    concat_and_savez_features_ = functools.partial(concat_and_savez_features, 
                                                    extractors=extractors,
                                                    sat_street_map=sat_street_map,
                                                    concat_save_path=data_path["concat"],
                                                    lock=lock)

    with ThreadPoolExecutor(max_workers=40) as concat_executor: 
        concat_futures = []
        for sat_fpath in sampled_sat_file_paths:
            concat_futures.append(concat_executor.submit(concat_and_savez_features_, 
                                            sat_fpath=sat_fpath))
                                            
        for concat_future in as_completed(concat_futures):
            print(concat_future.result())

    return f"Done aggregating features for: {dhs_id}"

if __name__ == "__main__":
    data_path = {}
    data_path["sat"] = "data/dhs_npzs_subset_has_street"
    data_path["street"] = "data/street_npzs_notna"
    data_path["concat"] = "data/concat_data"

    print("Constructing sat_street_map and dhsid_index_map")
    country_dir_locks = {}
    sat_street_map = {} # key = path to a satellite npz file, value = set of paths to its street files
    dhsid_index_map = {}
    for dataset_type in os.listdir(data_path["street"]): # train, test, val
        if dataset_type == "test" or dataset_type == "val":
            continue
        print(f"dataset_type: {dataset_type}")
        for country in os.listdir(os.path.join(data_path["street"], dataset_type)):
            print(f"country: {country}")
            if country not in country_dir_locks.keys():
                country_dir_locks[country] = threading.Lock()

            for street_fname in os.listdir(os.path.join(data_path["street"], dataset_type, country)):
                sat_index = street_fname.rsplit('-', 1)[0] # e.g. CD-2013-6#-00000003 in CD-2013-6#-00000003-1246492952431904.npz
                sat_fname = f"{sat_index}.npz"
                sat_file_path = os.path.join(data_path["sat"], dataset_type, country, sat_fname)
                concat_feat_fpath = os.path.join(data_path["concat"], dataset_type, country, sat_fname)

                # skip if we don't have data for this satellite image or if its aggregated data have been extracted
                if (not os.path.exists(sat_file_path)) or (os.path.exists(concat_feat_fpath)): 
                    continue

                street_file_path = os.path.join(data_path["street"], dataset_type, country, street_fname)
                if sat_file_path not in sat_street_map.keys():
                    sat_street_map[sat_file_path] = set([street_file_path])
                else:
                    sat_street_map[sat_file_path].add(street_file_path)

                dhs_id = sat_index.rsplit('-', 1)[0] # e.g. CD-2013-6# in e.g. CD-2013-6#-00000003
                if dhs_id not in dhsid_index_map.keys():
                    dhsid_index_map[dhs_id] = set([sat_file_path])
                else:
                    dhsid_index_map[dhs_id].add(sat_file_path)
                
    # print(f"sat_street_map: {sat_street_map}")
    print(f"dhsid_index_map keys: {dhsid_index_map.keys()}")

    sat_model_saved_path = "saved_model/satellite_resnet"
    street_model_saved_path = "saved_model/street_resnet"

    print("Loading saved models...")
    tf.device('/device:GPU:0')
    sat_model = tf.keras.models.load_model(sat_model_saved_path)
    street_model = tf.keras.models.load_model(street_model_saved_path)
    extractors = {}
    extractors['sat'] = Model(inputs=sat_model.inputs, outputs=sat_model.get_layer(index=-2).output)
    extractors['street'] = Model(inputs=street_model.inputs, outputs=street_model.get_layer(index=-2).output)
    # models['sat'] = "placeholder"
    # models['street'] = "placeholder"

    sample_size = 10000 # number of satellite images to sample per dhs survey id
    concat_and_savez_features_wrapper_ = functools.partial(concat_and_savez_features_wrapper,
                                            sample_size=sample_size,
                                            extractors=extractors,
                                            sat_street_map=sat_street_map,
                                            data_path=data_path)

    print("Aggregating satellite and street data and save aggregated data...")
    with ThreadPoolExecutor(max_workers=20) as executor: 
        futures = []
        for dhs_id, sat_file_paths in dhsid_index_map.items():
            country = dhs_id.split("-")[0]
            futures.append(executor.submit(concat_and_savez_features_wrapper_, 
                                            dhs_id=dhs_id, 
                                            sat_file_paths=sat_file_paths, 
                                            lock=country_dir_locks[country]))
        for future in as_completed(futures):
            print(future.result())

