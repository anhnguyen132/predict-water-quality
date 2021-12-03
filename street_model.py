import os
from os import listdir
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats
from tensorflow.keras import datasets, layers, models, losses, Model
from tqdm.auto import tqdm
from utils import load_dataset

def load_street_image(x_path, x):
  m = x.shape[0]
  for i in tqdm(range(0, m)):
    x[i] = np.load(x_path[i])["img_data"]

if __name__ == "__main__":
    prefix_path = "data/street_npzs_notna/"

    dataset_splits = {}
    dataset_splits["train"] = ["CD", "MD", "ZW", "CM", "GH", "NP"]
    dataset_splits["dev"] = ["BO", "BJ"]
    dataset_splits["test"] = ["AO", "AM"]

    X, Y = load_dataset(prefix_path, dataset_splits, model_img_size=[256, 256, 3], load_x=load_street_image) 
    train_x, dev_x, test_x = X
    train_y, dev_y, test_y = Y

    # Set up CNN model
    tf.device('/device:GPU:0')

    base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (256,256,3))
    for layer in base_model.layers:
        layer.trainable = False

    for i in range(45,50):
        base_model.layers[i].trainable = True  

    # Add additional layers to make the model to be regression
    model_temp = layers.Dense(64, activation='relu')(base_model)
    final_layer = layers.Dense(1, activation='sigmoid')(model_temp) 

    street_model = Model(inputs = model_temp_0.input, outputs = final_layer)

    street_model.compile(optimizer='adam',
                loss='mse',
                metrics=['mse'])

    checkpoint_path = "separate_training/street.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    print("Start training.")

    # Run the model
    history = street_model.fit(train_x,
                        train_y,
                        epochs=5,
                        batch_size=16,
                        validation_data=(dev_x, dev_y),
                        callbacks=[cp_callback, early_stop])    

    street_model.save('saved_model/street_resnet')
    print("Done saving the model. ")                         

    print("test labels: " + str(test_y_w[:10]))

    test_predictions = model_sat.predict(test_x)
    train_predictions = aggr_model.predict(train_x)
    dev_predictions = aggr_model.predict(dev_x)
  
    #########################################################################
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
    print("r_square result is " + str(r_square))

    mseObject = tf.keras.losses.MeanSquaredError()
    mseTensor = mseObject(test_y, predictions_flat)
    mse = mseTensor.numpy()

    print("normalized mse result is " + str(mse))

    mse_denorm = mseObject(test_y_w, predictions_flat_scaled).numpy()

    print("denormalized mse result is " + str(mse_denorm))

