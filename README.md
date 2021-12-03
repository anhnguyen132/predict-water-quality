# predict-water-quality
Predict water quality using multimodal inputs (Street level images and satellite images). Dataset and baseline model come from the [SustainBench](https://github.com/sustainlab-group/sustainbench.git) project. 

## Dataset
We choose to ignore examples without water quality index labels.
Due to compute power and storage constraints, we trained our model on a subset of the SustainBench dataset. 

### Satellite image data for satellite CNN model 
For each set of examples grouped by country and year, sample 100 examples that have water quality index label.

Total number of examples in SustainBench dataset for water quality index = 87,938

**Dataset split**

Uniform train/validation/test dataset split by country. Specifically, 

- Train set
    - 12 countries: TZ, BF, CM, GH, IA, KM, LS, ML, MW, NG, PH, TG
- Validation set
    - 11 countries: BJ, BO, CO, DR
- Test set 
    - 10 countries: AM, AO

Note: There are 7 countries with no water quality index label that we excluded. They are HN, ID, JO, KH, MA, MB, NI.

### Street-level images for street-level CNN model

**Dataset split**
Uniform train/validation/test dataset split by country. Specifically, 

- Train set: 
    - 6 countries: CD, MD, ZW, CM, GH, NP
- Validation set 
    - 2 countries: BJ, BO
- Test set 
    - 2 countries: AM, AO

Fewer countries were sampled because each satellite image has 0 to 100 street images which quickly exceeded our computational resources when we chose to use the same dataset split by country as for satellite images.

### Images for the aggregated model
We chose to ignore the satellite images that have no corresponding street-level images. 

Total number of examples sampled = 1,095

Uniform train/validation/test dataset split by country. Specifically, 

- Train set
    - 5 countries: CD, MD, ZW, CM, GH, NP, TJ, BD

- Validation set
    - 2 countries: BJ, BO

- Test set
    - 2 countries: AM, AO

## Files structure
- `aggr_model.py`, `satellite_customed_cnn.py`, `satellite_resnet.py`, `street_model.py`: model files to train, evaluate and run predictions
- `extract_features`: extracts features from satellite and street images (as the second to last layer in their respective trained CNN model), averages out the street-image feature vectors for each satellite feature vector and concatenates the averaged vector to the satellite feature vector.
- `split_dataset.sh`: script to split the country folders to train/dev/test sets
- `unzip_street_images.py`: unzips the street images which were zipped by country
- `preprocess_street.py`: upzip the street image files, filters out the images with invalid or missing labels, preprocesses the street images into dimensions 256 x 256 x 3, and saves these processed image data into npz files
- `utils.py`: helper functions to load and split the data sets and labels 


