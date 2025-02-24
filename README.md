# parkinson_speech
Classify speech as belonging to a Parkinson patient or not.

## Set up
This repository requires 3.12.X or higher.
```
python3 -m pip install -r requirements.txt
```

## Usage
The script src/main.py, to run the full processing and training of the model:
```
python3 main.py
```
You can set the following arguments:
- data_file: path to csv file with data (default='./in/pd_speech_features_AdS.csv')
- config_file: path to config file to use (default='config.yaml')
```
python main.py --data_file <path_to_a_data_file> --config_file <path_to_a_config_file> 
```

## Processing steps
Data will be loaded from file defined by the data_file argument. If header is multiple, levels will be dropped to simplify data processing. Because of this, any feature name given in the config file should only be the name of the feature at the lowest level. If a higher name is given, it will not be recognized.

### Split data
Data will be split in a train dataset and a test dataset. The size of the fraction from the original data used for the test dataset is defined by the 'test_size' param in the config file. \
In order to test the model ability to predict on a patient never seen before, the split will be made as to have all observations (several observations can come from a single observation, in this example 3 per patient) from one patient in a single dataset, e.g. train and test datasets do not share any patient in common. Test dataset and train dataset share the same class proportions. \
Once splitting is done, patient ids are erased to treat all observations as independent.

### Oversampling
Data is strongly imbalanced, in this example there is 3 times more sick patients than controls. To increase observations from the control group, a SMOTE approach is taken. This increases the number of observations from the minority class to match the number of observations from the majority class. Data augmentation is only applied to the train dataset. This allows for the test dataset to only be made of real data. \
This stage is performed after the splitting to avoid data leakage.

### Normalization
Normalization is made using Standard Scaler, which normalize data by subtracting the mean and dividing by the standard deviation. This is the most common and simplest way of normalizing data. It is applied to all features, except the one defined in 'binary_columns' param in config.yaml. Binary feature should not be normalized as these features are categorical and the numbers do not represent a measurement. In this example, the feature 'gender' is binary and is not normalized. Normalization is calaculated and applied on the train dataset, parameters calculated from train dataset are directly applied to normalize the test dataset, avoiding data leakage.

### Feature selection
A number of feature too great lead to a very slow training (in particular for heavy models) for little to no gain in performance (or even loss in performance). To avoid this a step of feature selection is made, using the SelectKBest algorithm. The number of feature selected is defined by the 'nb_of_features' param in config file (default=50). If an invalid number is given, all features will be kept. \
Feature selection is made based on the train dataset and then applied to the test dataset. SelectKBest computes (here) the ANOVA F-value on the data and keeps the K highest scoring features.

## Classification
3 models are trained to be compaired on the same datasets. Hyperparameter tuning has been loosely done by repeatedly train models with different hyperparameters. The models are a simple SVM, an AdaBoost classifier (with 100 estimators) and a MultiLayer Perceptron with 4 hidden layers (of respectively 512, 256, 128 and 64 neurons) and an early stopping.

### Results
Result from 3 models are displayed as the confusion matrix for each model and their respective classification report and save to the out folder.

## Testing
### Usage
```
pytest 
```