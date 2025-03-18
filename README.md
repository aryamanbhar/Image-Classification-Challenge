
# Image Classification with HistGradientBoostingClassifier

## Team Name: ML Crusaders

### Team Members:
- Seyyid Thaika
- Aryaman Bhardwaj

## Overview
This repository contains the implementation of an image classification task using machine learning. The solution involves data preprocessing, feature extraction, dimensionality reduction, and the application of a classifier model (HistGradientBoostingClassifier) to predict image labels from a provided dataset.

### Accuracy: 0.69 (Kaggle Submission)

## Table of Contents
1. [Dataset Analysis](#dataset-analysis)
2. [Classifier Exploration](#classifier-exploration)
3. [Final Solution Description](#final-solution-description)
4. [Code Structure](#code-structure)
5. [Dependencies](#dependencies)
6. [How to Run](#how-to-run)
7. [Results](#results)
8. [License](#license)

## Dataset Analysis
The dataset used for this project consists of images belonging to multiple categories. Here’s an analysis of the dataset:
- Total number of categories: [Insert number]
- Number of images per category: [Insert details]
- Visualizations: Examples from each category can be visualized below:

[Insert visualizations of one example per category]

## Classifier Exploration
We explored two different classifiers:
1. **HistGradientBoostingClassifier**: Achieved an accuracy of **0.69** on Kaggle.
2. **RandomForestClassifier**: Achieved an accuracy of **0.65**.

### Analysis:
- HistGradientBoostingClassifier performs better due to its ability to sequentially correct errors and capture complex relationships.
- RandomForestClassifier, by building independent decision trees, does not capture the patterns as effectively as gradient boosting.

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

## Dependencies
- Python 3.x
- **Libraries**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `opencv-python`
  - `scikit-image`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-classification.git
   cd image-classification
   ```
2. Run the preprocessing script to load and preprocess the data:
   ```bash
   python data_preprocessing.py
   ```
3. Train the model:
   ```bash
   python model_training.py
   ```
4. Make predictions:
   ```bash
   python predict.py
   ```

The predictions will be saved in a CSV file named **submit.csv**.

## Results
The final model achieved an accuracy of **0.69** on the test set, and the results were submitted to Kaggle.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
