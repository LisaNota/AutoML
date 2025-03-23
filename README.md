# AutoML

Specialized framework designed for binary classification tasks, focusing exclusively on numerical features. It is built with an emphasis on speed and efficiency, utilizing modern libraries and parallel computing for data preprocessing, feature selection, and ensemble model building.

## Key Features

- **Data Handling with Polars**  
  The framework uses the [Polars](https://pola.rs/) library for fast and efficient data loading, processing, and manipulation.

- **Missing Data Imputation**  
  Missing values in the dataset are filled with zeros (beta).

- **Feature Selection Using Random Forest**  
  For rapid initial feature selection, a Random Forest model is employed as the fastest method. This selection process is executed in parallel, quickly identifying the most informative features.

- **Model Training with Stepwise Stacking**  
  The core idea is to build an ensemble through a stepwise stacking algorithm:
  - Different subsets of features (e.g., top-25, top-50, top-75 most important features) are used for training.
  - CatBoost, XGBoost, Random Forest, and Logistic Regression are trained on these selected feature subsets.
  - First, a base model is trained and its predictions are computed using cross-validation. Then, additional models are iteratively added or removed—while optimizing their weights in the ensemble—mimicking the approach of stepwise regression. This process continues until the optimal combination of models is achieved.

