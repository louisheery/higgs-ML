# Higgs ML (Higgs boson Machine Learning Classifiers)

[![Build Status](https://img.shields.io/badge/build-v1.0-brightgreen)](https://github.com/louisheery/higgs-ML)
[![Build Status](https://img.shields.io/badge/build_status-published-brightgreen)](https://github.com/louisheery/plug-and-play-ML)

## File Contents  
### [Boosted Decision Trees](boosted-decision-tree-ml)
- [XGBoostBDT.py](boosted-decision-tree-ml/XGBoostBDT.py) – XGBoost BDT Classifier
- [AdaBoostBDT.py](boosted-decision-tree-ml/adaboostBDT.py) – AdaBoost Boosted Decision Tree Classifier
- [randomForestBDT.py](boosted-decision-tree-ml/randomForestBDT.py) – Random Forest Classifier
- [datasetCorrelationMatrices.py](boosted-decision-tree-ml/datasetCorrelationMatrices.py) – Plots correlation matrices of dataset of model's input variables
- [variationOfOptimalSensitivity.py](boosted-decision-tree-ml/variationOfOptimalSensitivity.py) – Determines optimal Input Variables that affect Classifier Sensitivity
- [XGBoostOptimisation.py](boosted-decision-tree-ml/XGBoostOptimisation.py) – Determines optimal hyperparameters of XGBoost Classifier
- [XGBoostStandardError.py](boosted-decision-tree-ml/XGBoostStandardError.py) – Calculates standard error of the XGBoost Classifier

## Summary of Results
### [Boosted Decision Trees](boosted-decision-tree-ml)
#### Optimal Boosted Decision Tree Output Plots
<img src="https://github.com/louisheery/higgs-ml/blob/master/boosted-decision-tree-ml/outputs/XGBoost_2Jets_200estimators_4depth_0.15learnrate.png" width="30%"><img src="https://github.com/louisheery/higgs-ml/blob/master/boosted-decision-tree-ml/outputs/XGBoost_3Jets_200estimators_4depth_0.15learnrate.png" width="30%">

#### Boosted Decision Tree Hyperparameter Optimisation
<img src="https://github.com/louisheery/higgs-ml/blob/master/boosted-decision-tree-ml/outputs/XGBoost_maxdepthnestimators.png" width="30%"><img src="https://github.com/louisheery/higgs-ml/blob/master/boosted-decision-tree-ml/outputs/XGBoost_maxdepthlearnrate.png" width="30%"><img src="https://github.com/louisheery/higgs-ml/blob/master/boosted-decision-tree-ml/outputs/XGBoost_samplesizelearnrate.png" width="30%">

#### Boosted Decision Tree Input Variable Optimisation
<img src="https://github.com/louisheery/higgs-ml/blob/master/boosted-decision-tree-ml/outputs/VariationOfOptimalSensitivity_2_Jets.png" width="30%"><img src="https://github.com/louisheery/higgs-ml/blob/master/boosted-decision-tree-ml/outputs/VariationOfOptimalSensitivity_3_Jets.png" width="30%"><img src="https://github.com/louisheery/higgs-ml/blob/master/boosted-decision-tree-ml/outputs/VariationOfOptimalSensitivity_Combined.png" width="30%">
