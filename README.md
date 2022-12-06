# BME590


# Machine Learning for Structural Analysis of Drugs that Permeate the Blood-Brain Barrier

## Created By: Christian Garvin, Kush Gulati, and Jamie Wang
***

We are interested in developing a qualitative model to classify whether a molecule is capable of crossing the blood-brain barrier (BBB) while improving our chemical understanding of what types of molecules can do so effectively.

# Our Workflow

## 1. Data Source
#### Adenot_BBB+.csv, list of permeable drugs and their SMILES
#### Adenot_BBB-.csv, list of non-permeable drugs and their SMILES
#### From: https://pubs.acs.org/doi/pdf/10.1021/ci034205d

## 2. Data Processing 
#### adenot.csv, combined and labeled BBB+/-
#### adenot_processing.ipynb
#### adenot_processed.csv

## 3. Data Visualization, Clustering, and Dimensionality Reduction
#### dim_red_feat_analysis.ipynb

## 4. Random Forest Model 
#### random_forest.ipynb

## 5. Support Vector Machine (SVM) Model 
#### svm_model_optimization.ipynb
#### svm_fragment_extraction.ipynb
#### svm_feature_extraction.ipynb

## 6. XGBoost Model
#### xgboost_optimization.ipynb
#### xgboost_fragment_extraction.ipynb
#### xgboost_feature_extraction.ipynb

## 7. Ensemble Model
#### ensemble_model.ipynb

## 8. Model Performance Direct Comparison
#### figures.ipynb

## 8. Final Feature and Fragment Analysis for SVM and XGBoost Models
#### dim_red_feat_analysis.ipynb







