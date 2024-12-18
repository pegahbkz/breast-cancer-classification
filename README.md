# Breast Cancer Classification

## About This Project
This project is an implementation of four classification models categorizing breast cancer instances into malignant and benign groups using the Breast Cancer Wisconsin Dataset.

## Built With
- pandas
- numpy
- sklearn
- matplotlib

## Dataset
The project uses the **Breast Cancer Wisconsin Dataset**. Ensure the `data.csv` file is in the root directory. The dataset includes:
- **Features**: Columns `2-31`
- **Labels**: Column `diagnosis` (`M` for Malignant, `B` for Benign)
- **ID column**: Removed before processing.

## Models Implemented
1. Decision Tree (DT)
2. Support Vector Classifier (SVC)
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (NB)

## Features
- Evaluates model performance with:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Optimizes KNN by testing `k=2` to `k=9`.
- Compares algorithm accuracy using visualizations.

## How to Run
### Prerequisites
- Python 3.x
- Required libraries:
  ```bash
  pip install pandas numpy scikit-learn matplotlib
