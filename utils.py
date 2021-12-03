import os
from os import listdir
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
from tqdm.auto import tqdm
import scipy.stats
# from dhs_labels import get_dhs_labels

# get DHS labels
def get_dhs_labels():
    dhs_labels_df = pd.read_csv("data/dhs_final_labels.csv")
    dhs_labels_df['survey'] = dhs_labels_df['DHSID_EA'].str[:10]
    dhs_labels_df['cc'] = dhs_labels_df['DHSID_EA'].str[:2]
    dhs_labels_df['path'] = 'dhs_npzs/' + dhs_labels_df['survey'] + '/' + dhs_labels_df['DHSID_EA'] + '.npz'
    dhs_labels_df.set_index('DHSID_EA', verify_integrity=True, inplace=True)

    return dhs_labels_df

def create_data_path(prefix_path, set_type, dataset, sample_num=None):
  x_path = []
  x_index = []
  for country in dataset:
    num = 0
    for file_name in listdir(os.path.join(prefix_path, set_type, country)):
      x_path.append(os.path.join(prefix_path, set_type, country, file_name))
      x_index.append(file_name[:-4]) # get AL-2008-5#-00000115 in AL-2008-5#-00000115.npz
      num += 1
      if (sample_num is not None) and (num == sample_num): 
        break
  return x_path, x_index

# get the water labels
def get_y_w(df_labels, x_index):
  y = []
  for i in range (0, len(x_index)):
    y.append(df_labels.loc[x_index[i]]['water_index'])
  return y

# Change imgs of shape 8x255x255 to 255x255x8
def format_input_img(imgs):
    new_imgs = np.empty([imgs.shape[0], imgs.shape[2], imgs.shape[3], imgs.shape[1]], dtype=imgs.dtype)
    for i in range(0, imgs.shape[0]):
        img = imgs[i, :, :, :]
        for j in range(0, imgs.shape[1]):
            band = np.reshape(img[j, :, :], img[0].shape + (1,))
            if (j == 0):
                new_img = band;
            else:
                new_img = np.concatenate((new_img, band), axis=new_img.ndim-1)

        new_imgs[i] = new_img 

    return new_imgs

def calculate_r_square(input, output):
    if (np.all(output == output[0])):
      output[0] = output[0] - 1e-7

def load_dataset(prefix_path, dataset_splits, model_img_size, load_x):
    
    sample_nums = {}
    if prefix == "data/street_npzs_notna/":
        sample_nums['train'] = 2000
        sample_nums['dev'] = 1000
        sample_nums['test'] = 500
    else: 
        sample_nums['train'] = None
        sample_nums['dev'] = None
        sample_nums['test'] = None

    train_x_path, train_x_index = create_data_path(prefix_path, "train", dataset_splits["train"], sample_nums['train'])
    dev_x_path, dev_x_index = create_data_path(prefix_path, "val", dataset_splits["dev"], sample_nums['dev'])
    test_x_path, test_x_index = create_data_path(prefix_path, "test", dataset_splits["test"], sample_nums['test'])

    train_x = np.empty([len(train_x_path)] + model_img_size)
    dev_x = np.empty([len(dev_x_path)] + model_img_size)
    test_x = np.empty([len(test_x_path)] + model_img_size)

    print("Importing data...")

    print("train set")
    load_x(train_x_path, train_x)
    print("dev set")
    load_x(dev_x_path, dev_x)
    print("test set")
    load_x(test_x_path, test_x)

    if prefix_path == "data/dhs_npzs_subset_notna/": # format satellite data from 8 x 255 x 255 to 255 x 255 x 8
        train_x = format_input_img(train_x)
        dev_x = format_input_img(dev_x)
        test_x = format_input_img(test_x)

    # normalize
    train_x, dev_x, test_x = train_x / 255.0, dev_x / 255.0, test_x / 255.0
    print("size of train_x: " + str(train_x.shape))
    print("size of dev_x: " + str(dev_x.shape))
    print("size of test_x: " + str(test_x.shape))
  
    # train_y, dev_y, test_y = None, None, None
    # Get DHS labels
    df_labels = get_dhs_labels()

    # if df_labels is not None:
    train_y_w = get_y_w(df_labels, train_x_index)
    dev_y_w = get_y_w(df_labels, dev_x_index)
    test_y_w = get_y_w(df_labels, test_x_index)

    train_y = np.asarray(train_y_w) 
    dev_y = np.asarray(dev_y_w)
    test_y = np.asarray(test_y_w)

    #normalize
    train_y = (train_y - 1) / 4
    dev_y = (dev_y - 1) / 4
    test_y = (test_y - 1) / 4

    print("size of train_y: " + str(train_y.shape))
    print("size of dev_y: " + str(dev_y.shape))
    print("size of test_y: " + str(test_y.shape))

    print("Done!")

    return (train_x, dev_x, test_x), (train_y, dev_y, test_y)

