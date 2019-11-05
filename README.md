# Higgs ML (Higgs boson Machine Learning Classifiers)

[![Build Status](https://img.shields.io/badge/build-v1.0-brightgreen)](https://github.com/louisheery/higgs-ML)
[![Build Status](https://img.shields.io/badge/build_status-published-brightgreen)](https://github.com/louisheery/plug-and-play-ML)

Supervised Machine Learning techniques used to categorise VH, H(bb) Higgs boson decay events using data collected from the Large Hadron Collider, CERN.

![alt text](https://github.com/louisheery/higgs-ml/blob/master/higgs-ml-screenshots.png)

## Repo Contents  
### [1. Boosted Decision Trees](boosted-decision-tree-ml)
#### a) Different BDT Classifiers
- [XGBoostBDT.py](boosted-decision-tree-ml/XGBoostBDT.py) – XGBoost BDT Classifier
- [AdaBoostBDT.py](boosted-decision-tree-ml/adaboostBDT.py) – AdaBoost Boosted Decision Tree Classifier
- [randomForestBDT.py](boosted-decision-tree-ml/randomForestBDT.py) – Random Forest Classifier
#### b) XGBoost Optimisation
- [variationOfOptimalSensitivity.py](boosted-decision-tree-ml/variationOfOptimalSensitivity.py) – Determines optimal Input Variables that affect Classifier Sensitivity
- [XGBoostOptimisation.py](boosted-decision-tree-ml/XGBoostOptimisation.py) – Determines optimal hyperparameters of XGBoost Classifier
- [XGBoostStandardError.py](boosted-decision-tree-ml/XGBoostStandardError.py) – Calculates standard error of the XGBoost Classifier
#### c) Dataset Analysis
- [datasetCorrelationMatrices.py](boosted-decision-tree-ml/datasetCorrelationMatrices.py) – Plots correlation matrices of dataset of model's input variables

### [2. Deep Neural Networks](deep-neural-network-ml)
- [DNN.py](deep-neural-network-ml/DNN.py) – DNN Classifier
- [DNNOptimisation.py](deep-neural-network-ml/DNNOptimisation.py) – Determines Optimal Hyperparameters of DNN Classifier
- [DNNStandardError.py](deep-neural-network-ml/DNNStandardError.py) – Calculates Standard Error of the DNN Classifier

### [3. Adversarial Neural Networks](adversarial-nn-ml)
#### a) Dijet-mass Aware ANN
- [dijetmassANN.py](adversarial-nn-ml/dijetmassANN.py) – Dijet-mass Aware Adversarial Neural Network Classifier
- [dijetmassANN_withmBBDistributionGraphs.py](adversarial-nn-ml/dijetmassANN_withmBBDistributionGraphs.py) – Dijet-mass Aware Adversarial Neural Network Classifier with Graphs
#### b) Monte Carlo Generator Aware ANN
- [generatorANN.py](adversarial-nn-ml/generatorANN.py) – Monte Carlo Event Generator Aware Adversarial Neural Network Classifier
- [generatorANN_withmBBDistributionGraphs.py](adversarial-nn-ml/generatorANN_withmBBDistributionGraphs.py) – Monte Carlo Event Generator Aware Adversarial Neural Network Classifier with Graphs

### [4. Dataset & Plotting](dataset-and-plotting)
#### a) H(bb) Monte Carlo Generated Event Dataset
- [CSV](dataset-and-plotting/CSV) – Original Dataset.
Used for training [Boosted Decision Tree Classifiers](boosted-decision-tree-ml/XGBoostBDT.py) and [Deep Neural Network Classifiers](deep-neural-network-ml/DNN.py).
- [CSV_withBinnedDijetMassValues](dataset-and-plotting/CSV_withBinnedDijetMassValues) – Original Dataset, with added column of data which bins the raw dijet mass values into 10 equally sized bins (labelled 1 to 10).
Used for training [Dijet mass-aware Adversarial Neural Network Classifiers](adversarial-nn-ml/dijetmassANN.py).
- [CSV_differentGenerators](dataset-and-plotting/CSV_differentGenerators) – Original Dataset, which has been duplicated and a non-linear distortion applied to one of the copies, thus simulating Data originating from different Monte Carlo Generators.
Used for training [Generator-aware Adversarial Neural Networks Classifiers](adversarial-nn-ml/generatorANN.py).

#### b) Graph Plotting Files
- [bdtPlotting.py](dataset-and-plotting/bdtPlotting.py) – Scripts used to plot Boosted Decision Tree (BDT) Output Graphs.
- [nnPlotting.py](dataset-and-plotting/nnPlotting.py) – Scripts used to plot Neural Network (NN) and Adversarial Neural Network (ANN) Output Graphs.
- [sensitivity.py](dataset-and-plotting/sensitivity.py) – Scripts used to calculate sensitivity of BDT and NN and ANN Classifiers.


## Download Repo
- Download Link is [HERE](https://github.com/louisheery/higgs-ML/archive/master.zip)

- Fork Repo using:
```
$ git clone https://github.com/louisheery/higgs-ML.git
```
