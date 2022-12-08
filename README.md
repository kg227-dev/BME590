# BME590


# Machine Learning for Structural Analysis of Drugs that Permeate the Blood-Brain Barrier

## Created By: Christian Garvin, Kush Gulati, and Jamie Wang
***

We are interested in developing a qualitative model to classify whether a molecule is capable of crossing the blood-brain barrier (BBB) while improving our chemical understanding of what types of molecules can do so effectively.

# Our Workflow

## 1. Data Source
___
#### Adenot_BBB+.csv, list of permeable drugs and their SMILES
#### Adenot_BBB-.csv, list of non-permeable drugs and their SMILES
#### From: https://pubs.acs.org/doi/pdf/10.1021/ci034205d

## 2. Data Processing 
___
#### adenot_processing.ipynb
#### adenot.csv, combined and labeled BBB+/-
#### adenot_processed.csv
Perform data processing steps such as: Labeling, deleting missing data, clean and wash SMILES, remove duplicates.

## 3. Data Visualization, Clustering, and Dimensionality Reduction
___
#### dim_red_feat_analysis.ipynb
Performed K-Means clustering (k=5) to assist in visualizing clustered data. For visualization, performed principal component analysis and t-SNE, and plotted data class distribution. 

## 4. Random Forest Model
___
#### random_forest.ipynb
Built a Random Forest Classifier hyperparametrized on n_estimators=100, criterion="gini", and max_depth=20.  Evaluated the model on a random split and scaffold split on "adenot_processed.csv", using accuracy, balanced accuracy, ROC AUC, precision, and recall. 
<br />  <br /> __Random Split__
| Metric        | Score           | 
| ------------- |-------------| 
| Accuracy      | 0.9538152610441767 |
| Balanced Accuracy     | 0.8826530612244898      |  
| ROC AUC |     0.8826530612244897  |   
| Precision | 0.9456264775413712    |  
| Recall | 1.0      |  

<br /> __Scaffold Split__
| Metric        | Score           | 
| ------------- |-------------| 
| Accuracy      | 0.8012048192771084 |
| Balanced Accuracy     | 0.777027027027027     |  
| ROC AUC |     0.777027027027027  |   
| Precision | 0.736    |  
| Recall | 1.0      |  


## 5. Support Vector Machine (SVM) Model 
___
#### svm_model_optimization.ipynb
Built a Support Vector Machine Classifier hyperparametrized on C=0.1, gamma=1, kernel="linear".  Evaluated the model on a random split and scaffold split on "adenot_processed.csv", using accuracy, balanced accuracy, ROC AUC, precision, and recall. 
<br />  <br /> __Random Split__
| Metric        | Score           | 
| ------------- |-------------| 
| Accuracy      | 0.963855421686747 |
| Balanced Accuracy     | 0.9158673469387755     |  
| ROC AUC |     0.9158673469387756  |   
| Precision | 0.961352657004831    |  
| Recall | 0.995      |  

<br /> __Scaffold Split__
| Metric        | Score           | 
| ------------- |-------------| 
| Accuracy      | 0.8644578313253012 |
| Balanced Accuracy     | 0.8499559341950647     |  
| ROC AUC |     0.8499559341950647  |   
| Precision | 0.8116591928251121    |  
| Recall | 0.9836956521739131      |  
#### svm_feature_extraction.ipynb
Extracted feature importances of tuned SVM and displayed the top 10 features with the highest importances. 

## 6. XGBoost Model
___
#### xgboost_optimization.ipynb
Built a XGBoost Classifier tuned on learning_rate=0.1, n_estimators=1000, max_depth=4, min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.85, reg_alpha=1e-05, objective="binary:logistic", nthread=4, and scale_pos_weight=1. Evaluated the model on a random split and scaffold split on "adenot_processed.csv", using accuracy, balanced accuracy, ROC AUC, precision, and recall. 
<br />  <br /> __Random Split__
| Metric        | Score           | 
| ------------- |-------------| 
| Accuracy      | 0.9558232931726908 |
| Balanced Accuracy     | 0.9070153061224491     |  
| ROC AUC |     0.9070153061224491  |   
| Precision | 0.9587378640776699    |  
| Recall | 0.9875      |  

<br /> __Scaffold Split__
| Metric        | Score           | 
| ------------- |-------------| 
| Accuracy      | 0.8674698795180723 |
| Balanced Accuracy     | 0.8520123384253819     |  
| ROC AUC |     0.8520123384253818  |   
| Precision | 0.8097345132743363    |  
| Recall | 0.9945652173913043      |  
#### xgboost_feature_extraction.ipynb
Extracted feature importances of tuned XGBoost and displayed the top 10 features with the highest importances. 


## 7. Ensemble Model
___
#### ensemble_model.ipynb
Built ensemble voting classifier with tuned SVM, XGBoost, and Random Forest classifiers and equal voting.  Used soft voting, which involves summing the predicted probabilities for class labels and predicting the class label with the largest sum probability. Evaluated the model on a random split and scaffold split on "adenot_processed.csv", using accuracy, balanced accuracy, ROC AUC, precision, and recall. 
<br />  <br /> __Random Split__
| Metric        | Score           | 
| ------------- |-------------| 
| Accuracy      | 0.9618473895582329 |
| Balanced Accuracy     | 0.910765306122449     |  
| ROC AUC |     0.910765306122449  |   
| Precision | 0.9590361445783132    |  
| Recall | 0.995      |  

<br /> __Scaffold Split__
| Metric        | Score           | 
| ------------- |-------------| 
| Accuracy      | 0.8614457831325302 |
| Balanced Accuracy     | 0.8445945945945945     |  
| ROC AUC |     0.8445945945945946  |   
| Precision | 0.8    |  
| Recall | 1.0      | 



## 8. Model Performance Direct Comparison
___
#### figures.ipynb
Compiled the performance metrics from each notebook and created bar plots and radar plots. 

## 8. Final Feature and Fragment Analysis for SVM and XGBoost Models
___
#### dim_red_feat_analysis.ipynb
For both SVM and XGBoost Models:
1. Examined percentage of permeable/non-permeable classes with respective top features present
    Permeable molecules were highly represented by top SVM model features
    Almost only non-permeable molecules were highly represented by top XGBoost model features
3. Examined percentage of clusters with respective top features present
    Cluster 0 was highly represented by top XGBoost model features
    Cluster 3 was highly represented by top SVM model features

Performed example fragment analysis:
1. Cluster 0 (beta-lactam antibiotic) fragment analysis based on top features from the XGBoost model
2. Cluster 3 (corticosteroid) fragment analysis based on top features from the SVM model







