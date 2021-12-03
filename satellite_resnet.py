import os
from os import listdir
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
from tqdm.auto import tqdm
import scipy.stats
from utils import get_dhs_labels, calculate_r_square, load_dataset

def load_satellite_image(input_path, input_x):
  m = input_x.shape[0]
  for i in tqdm(range(0, m)):
    image = np.load(input_path[i])['x']
    for j in range(0,8):
      input_x[i][j] = image[j]

if __name__ == "__main__":
  prefix_path = "data/dhs_npzs_subset_notna/"

  dataset_splits = {}
  dataset_splits["train"] = ["TZ","BF","CM","GH","IA","KM","LS","ML","MW","NG","PH","TG"]
  dataset_splits["dev"] = ["BJ", "BO", "CO", "DR"]
  dataset_splits["test"]  = ["AM", "AO"]

  X, Y = load_dataset(prefix_path, dataset_splits, model_img_size=[8, 255, 255], load_x=load_satellite_image) 
  train_x, dev_x, test_x = X
  train_y, dev_y, test_y = Y

  # Set up CNN model
  tf.device('/device:GPU:0')

  base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (51,51,3))
  for layer in base_model.layers:
    layer.trainable = False

  model_temp_0 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(3, (8, 8), activation='relu', input_shape=(255, 255, 8)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(3, (8, 8), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(3, (8, 8), activation='relu'),
  ])

  for i in range(45,50):
    base_model.layers[i].trainable = True  

  # Add additional layers to make the model to be regression
  model_temp_1 = base_model(model_temp_0.output)
  model_temp_2 = layers.Flatten()(model_temp_1)
  model_temp_3 = layers.Dense(64, activation='relu')(model_temp_2)
  final_layer = layers.Dense(1, activation='sigmoid')(model_temp_3) 

  model_sat = Model(inputs = model_temp_0.input, outputs = final_layer)

  model_sat.compile(optimizer='adam', loss='mse', metrics=['mse'])

  checkpoint_path = "separate_training/satellite_resnet.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
  early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

  print("Start training.")

  # Run the model
  history = model_sat.fit(train_x,
                      train_y,
                      epochs=10,
                      batch_size=16,
                      validation_data=(dev_x, dev_y),
                      callbacks=[cp_callback, early_stop])  

  model_sat.save('saved_model/satellite_resnet')
  print("Done saving the model. ")                       

  print("test labels: " + str(test_y[:10]*4 + 1))
  print("normalized test labels: " + str(test_y[:10]))

  test_predictions = model_sat.predict(test_x)
  train_predictions = aggr_model.predict(train_x)
  dev_predictions = aggr_model.predict(dev_x)
  
  predictions_flat = test_predictions.flatten()
  predictions_flat_scaled = predictions_flat * 4 + 1
  #########################################################################
  print("==== Train ====")
  print(mseObject(train_y, train_predictions.flatten()).numpy())

  print("==== Dev ====")
  print(mseObject(dev_y, dev_predictions.flatten()).numpy())
  
  print("==== Test ====")
  print("predictions: " + str(predictions_flat_scaled[:10]))
  print("normalized predictions: " + str(predictions_flat[:10]))

  
  r_square = scipy.stats.pearsonr(test_y, predictions_flat)[0]**2
  print("r_square result is " + str(r_square))

  mseObject = tf.keras.losses.MeanSquaredError()
  mseTensor = mseObject(test_y, predictions_flat)
  mse = mseTensor.numpy()

  print("normalized mse result is " + str(mse))

  mse_denorm = mseObject(test_y*4 + 1, predictions_flat_scaled).numpy()

  print("denormalized mse result is " + str(mse_denorm))