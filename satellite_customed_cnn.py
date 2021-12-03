import os
from os import listdir
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats

# Get DHS labels
df = pd.read_csv("data/dhs_final_labels.csv")
df['survey'] = df['DHSID_EA'].str[:10]
df['cc'] = df['DHSID_EA'].str[:2]
df['path'] = 'data/npzs/AL_DR/' + df['DHSID_EA'] + '.npz'

df.set_index('DHSID_EA', verify_integrity=True, inplace=True)

# Split data
prefix_path = "data/dhs_npzs_subset_notna/"

train_set = ["TZ","BF","CM","GH","IA","KM","LS","ML","MW","NG"]
#,"PH","TG","TZ","ZM","BD","CD","ET","GU","KE","LB", "MD", "MM", "MZ","PE","SN","TJ","UG","ZW"]
dev_set = ["BJ", "BO", "CO", "DR"]
# "GA", "GN", "GY", "HT", "NM", "SL", "TD"]
test_set = ["AM", "AO"]
# , "CI", "EG", "KY", "NP",  "PK", "RW", "SZ"]


def create_data_path(set_type, set):
  x_path = []
  x_index = []
  countries = []
  for i in range(0, len(set)):
    countries.append(prefix_path + set_type + "/" + set[i] + "/")
  for country in countries:
    for file in listdir(country):
      x_path.append(country + file)
      x_index.append(file[:-4])
  return x_path, x_index

train_x_path, train_x_index = create_data_path("train", train_set)
dev_x_path, dev_x_index = create_data_path("val", dev_set)
test_x_path, test_x_index = create_data_path("test", test_set)

def get_y_w(x_index):
  y = []
  for i in range (0, len(x_index)):
    y.append(df.loc[x_index[i]]['water_index'])
  return y

train_y_w = get_y_w(train_x_index)
dev_y_w = get_y_w(dev_x_index)
test_y_w = get_y_w(test_x_index)

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

def load_satellite_image(input_path, input, m):
  for i in range(0, m):
    image = np.load(input_path[i])['x']
    for j in range(0,8):
      input[i][j] = image[j]

train_x_t = np.empty([len(train_x_path), 8, 255, 255])
dev_x_t = np.empty([len(dev_x_path), 8, 255, 255])
test_x_t = np.empty([len(test_x_path), 8, 255, 255])

print("Importing satellite data...")

load_satellite_image(train_x_path, train_x_t, len(train_x_path))
load_satellite_image(dev_x_path, dev_x_t, len(dev_x_path))
load_satellite_image(test_x_path, test_x_t, len(test_x_path))

train_x = format_input_img(train_x_t)
train_y = np.asarray(train_y_w)
dev_x = format_input_img(dev_x_t)
dev_y = np.asarray(dev_y_w)
test_x = format_input_img(test_x_t)
test_y = np.asarray(test_y_w)

train_x, dev_x, test_x = train_x / 255.0, dev_x / 255.0, test_x / 255.0
train_y = (train_y - 1) / 4
dev_y = (dev_y - 1) / 4
test_y = (test_y - 1) / 4
 
print("Done!")

print("size of train_x: " + str(train_x.shape))
print("size of train_y: " + str(train_y.shape))
print("size of dev_x: " + str(dev_x.shape))
print("size of dev_y: " + str(dev_y.shape))
print("size of test_x: " + str(test_x.shape))
print("size of test_y: " + str(test_y.shape))

# Set up CNN model
tf.device('/device:GPU:0')

def create_model_sat():
  model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(255, (8, 8), activation='relu', input_shape=(255, 255, 8)),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(255, (8, 8), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(255, (8, 8), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='sigmoid'),
  tf.keras.layers.Dense(1)
])
  return model

model_sat = create_model_sat()

model_sat.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

checkpoint_path = "separate_training/satellite_cnn.ckpt"
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

print("test labels: " + str(test_y_w[:10]))
print("normalized test labels: " + str(test_y[:10]))

predictions = model_sat.predict(test_x)
predictions_flat = predictions.flatten()
predictions_flat_scaled = predictions_flat * 4 + 1
print("predictions: " + str(predictions_flat_scaled[:10]))
print("normalized predictions: " + str(predictions_flat[:10]))

r_square = scipy.stats.pearsonr(test_y_w, predictions_flat_scaled)[0]**2
print("r_square result is " + str(r_square))

normalized_r_square = scipy.stats.pearsonr(test_y, predictions_flat)[0]**2

print("r_square result on normalized labels and predictions: " + str(normalized_r_square))

mseObject = tf.keras.losses.MeanSquaredError()
mseTensor = mseObject(test_y, predictions_flat)
mse = mseTensor.numpy()

print("normalized mse result is " + str(mse))

mse_denorm = mseObject(test_y_w, predictions_flat_scaled).numpy()

print("denormalized mse result is " + str(mse_denorm))