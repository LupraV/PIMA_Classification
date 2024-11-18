# PIMA_Classification
PIMA Diabetes - Binary Classification (SVM, MLP, Logistic Regression + PCA/LDA)


## Introduction

In this data science project, I applied supervised machine learning techniques to predict diabetes based on medical data, focusing on classification models. The workflow includes data exploration, data wrangling, feature selection, and model evaluation. The goal was to accurately predict diabetes using models like SVM, MLP, Logistic Regression, and versions with PCA and LDA dimensionality reduction techniques.


#### Dataset used

The dataset utilized in this project is the Pima Indians Diabetes dataset.


### Research Questions

1. How effective are different imputation strategies for handling missing data?
2. Analyse the impact of normalisation and feature selection on the classification models
3. Compute and evaluate the models' performance at classifying diabetes
4. Which model among Logistic Regression, SVM, MLP, LR-PCA, and LR-LDA offers the best performance in predicting diabetes?


### Summary

This project involved extensive analysis and model development on a diabetes dataset, focusing on evaluating several machine learning models, including MLP, SVM, Logistic Regression, and Logistic Regression with PCA and LDA. Below is a summary of the key steps and findings:

1. **Dataset Exploration and Quality Checks**:

The dataset consists of 768 rows and 9 columns, with 8 features and a binary outcome indicating the presence or absence of diabetes.
Initial data cleaning addressed zero values in specific columns that were treated as missing data and converted to NaN.
The extreme SkinThickness outlier, was converted to missing values for subsequent imputation.

2. **Data Imputation**:

Missing values were imputed using the median, which was found to be more effective than KNN imputation. Median imputation based on class was also tested but did not render better classification outputs.

3. **Normalisation and Feature Selection**:

Data was normalized using StandardScaler, MinMaxScaler and RobustScaler ,but the MinMaxScaler outperformed the others overall, thus used throughout to ensure uniformity in feature scales.
Feature selection was applied, tested with the SelectFromModel and SelectKBest methods, to identify the most impactful features for each model.
SelectFromModel was providing better results on basically all models, thus only showing SelectKBest form the Logistic Regression models.

4. **Modeling and Evaluation**:

Using a stratified split of 80% train and 20% test (composed of only complete cases to avoid data leakage) and cross-validation, these were the most accurate models:

**MLP**: The best MLP model achieved an accuracy of 82% with an alpha parameter of 1.323 (GridSearchCV 10-fold, 30 iterations), activation: tanh,  solver: adam, and using SelectFromModel. Highest accuracy model, same as the optimised SVM.

**SVM**: The best SVM model using a RBF kernel and C=1 also achieved an accuracy of 82%, using features: Glucose, BMI, Age. Outperformed all other models classifying diabetes (class 1) with a f1-score of 0.73. Overall the model I would choose given the best balance of metrics, especially f1-score and recall.

**Logistic Regression**: Logistic Regression with SelectFromModel and MinMaxScaler achieved an accuracy of 82%.

**Logistic Regression with PCA** and SelectKBest had a similar accuracy (80%) but slightly better precision for class 1 (diabetic).

**Logistic Regression with FLD/LDA** and SelectFromModel and SelectKBest both achieved an accuracy of 77%, with consistent performance across metrics.

5. **Conclusion**: Overall the model I would choose given the best balance of metrics, especially f1-score and recall would be the optimised SVM.
