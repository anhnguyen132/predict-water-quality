import os
from os import listdir
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
from tqdm.auto import tqdm
import scipy.stats
from dhs_labels import get_dhs_labels
from utils import load_dataset

def load_concat_data(input_path, input_x):
  m = input_x.shape[0]
  for i in tqdm(range(0, m)):
    input_x[i] = np.load(input_path[i])['x']

if __name__ == "__main__":
  # Get DHS labels
  df_labels = get_dhs_labels()

  # Split data
  prefix_path = "data/concat_data/"

  dataset_splits = {}
  dataset_splits["train"] = ["CD", "MD", "ZW", "CM", "GH", "NP", "TJ", "BD"]
  dataset_splits["dev"] = ["BO", "BJ"]
  dataset_splits["test"] = ["AO", "AM"]

  X, Y = load_dataset(prefix_path, dataset_splits, model_img_size=[128], load_x=load_concat_data) 
  train_x, dev_x, test_x = X
  train_y, dev_y, test_y = Y

  # Set up CNN model
  tf.device('/device:GPU:0')

##################### Modifications for aggregation model #####################

  def create_aggr_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(64, activation='relu', input_shape=(128,)),
      tf.keras.layers.Dense(32, activation='relu', input_shape=(64,)),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

  aggr_model = create_aggr_model()

  aggr_model.compile(optimizer='adam', loss='mse', metrics=['mse'])

  early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

  print("Start training.")

  # Run the model
  history = aggr_model.fit(train_x,
                      train_y,
                      epochs=10,
                      batch_size=32,
                      validation_data=(dev_x, dev_y),
                      callbacks=[early_stop])                    

  print("test labels: " + str(test_y[:10]*4 + 1))
  print("normalized test labels: " + str(test_y[:10]))

  test_predictions = aggr_model.predict(test_x)
  train_predictions = aggr_model.predict(train_x)
  dev_predictions = aggr_model.predict(dev_x)
  ##############################################################################
  print("==== Train ====")
  print(mseObject(train_y, train_predictions.flatten()).numpy())

  print("==== Dev ====")
  print(mseObject(dev_y, dev_predictions.flatten()).numpy())
  
  print("==== Test ====")
  predictions_flat = test_predictions.flatten()
  predictions_flat_scaled = predictions_flat * 4 + 1
  print("predictions: " + str(predictions_flat_scaled[:10]))
  print("normalized predictions: " + str(predictions_flat[:10]))

  
  r_square = scipy.stats.pearsonr(test_y, predictions_flat)[0]**2
  print("test r_square result is " + str(r_square))

  mseObject = tf.keras.losses.MeanSquaredError()
  mseTensor = mseObject(test_y, predictions_flat)
  mse = mseTensor.numpy()

  print("normalized test mse result is " + str(mse))

  mse_denorm = mseObject(test_y*4 + 1, predictions_flat_scaled).numpy()

  print("denormalized test mse result is " + str(mse_denorm))