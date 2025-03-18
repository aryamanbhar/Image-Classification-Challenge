
# Image Classification with HistGradientBoostingClassifier

## Overview
This repository contains the implementation of an image classification task using machine learning for the Image Classification Challenge on Kaggle. The solution involves data preprocessing, feature extraction, dimensionality reduction, and the application of a classifier model (HistGradientBoostingClassifier) to predict image labels from a provided dataset.

### Accuracy: 0.69 (Kaggle Submission)

## Final Solution Description

### 1. Data Preprocessing
- **Data Loading**: Images are loaded and flattened into one-dimensional arrays.
- **Feature Extraction**:
  - **Gray Level Co-occurrence Matrix (GLCM)**: Extracts textural features.
  - **Edge Detection**: Uses the Canny algorithm.
  - **Corner Detection**: Features generated via the Harris corner detection algorithm.
  - **Histogram of Oriented Gradients (HOG)**: Captures shape and structural details.
  
  All these features are combined with raw pixel data to create a comprehensive representation of each image.

### 2. Feature Standardization
- All extracted features are standardized using `StandardScaler` to ensure uniform scaling for subsequent model training.

### 3. Dimensionality Reduction
- PCA (Principal Component Analysis) is applied to reduce the features to three principal components, retaining essential variance while reducing computational complexity.

### 4. Model Training
- The **HistGradientBoostingClassifier** is chosen as the final model, as it provides the best accuracy.
- The model is trained on the combined feature set and tested on the validation dataset.

### 5. Prediction and Output
- The trained model predicts labels for the test dataset and saves the results in a CSV file called **submit.csv**.

## Code Structure

- `data_preprocessing.py`: Code for loading data, feature extraction, and preprocessing.
- `model_training.py`: Code for training and evaluating models.
- `predict.py`: Code for making predictions and saving the output.
- `PCA_dimensionality_reduction.py`: Code for applying PCA to reduce the feature space.

The predictions are saved in a CSV file named **submit.csv**.

## Results
The final model achieved an accuracy of **0.69** on the test set, and the results were submitted to Kaggle.

