# Hyperparameter optimisation script which finds the optimal hyperparameters to train the XGBoost BDT with
# Author: Louis Heery

# How to Use:
# 1. [LINE 48] Replace 'hyperparameterOne = ' with the desired Hyperparameter from: maxDepth, nEstimators, learningRate, subSample
# 2. [LINE 49] Replace 'hyperparameterTwo = ' with the desired Hyperparameter from: maxDepth, nEstimators, learningRate, subSample
# 3. [LINE 75-78] Assign hyperparameterOneValue & hyperparameterTwoValue to their correct hyperparameter, and set the other two hyperparameters to their default value.
# 4. [LINE 84-87] Assign hyperparameterOneValue & hyperparameterTwoValue to their correct hyperparameter, and set the other two hyperparameters to their default value.

import pandas
import numpy
import sys
sys.path.append("../")
sys.path.append("../plotting/")
import pickle
import time
import threading

import matplotlib.cm as cm
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ../bdtPlotting import *
from ../sensitivity import * # sensitivity.py file, which has "calc_sensitivity_with_error" Function in it
from xgboost import XGBClassifier
from IPython.display import display
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivitB = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivitB,totalError)

maxDepth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
nEstimators = [1, 5, 20, 50, 100, 250, 500]
learningRate = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
sampleSize = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

hyperparameterOne = maxDepth
hyperparameterTwo = nEstimators

hyperparameterOneName = "MaxDepth"
hyperparameterTwoName = "Number of Estimators"

numberOfIterations = len(hyperparameterOne) * len(hyperparameterTwo)

dataset = np.zeros((numberOfIterations, 9))

i = 0

for hyperparameterOneValue in hyperparameterOne:
    for hyperparameterTwoValue in hyperparameterTwo:

        print("Training Model " + str(i + 1) + "/" + str(numberOfIterations) + " with Hyperparameters of")
        print("Hyperparameter One = ", hyperparameterOneValue)
        print("Hyperparameter Two = ", hyperparameterTwoValue)

        start = time.time()

        for nJets in [2,3]:

            # Defining BDT Parameters
            if nJets == 2:
                variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]
                n_estimators = hyperparameterTwoValue # Default = 200
                max_depth = hyperparameterOneValue # Default = 4
                learning_rate = 0.15 # Default = 0.15
                subsample = 0.5 # Default = 0.5

            else:
                variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont','nTrackJetsOR',]
                n_estimators = hyperparameterTwoValue # Default = 200
                max_depth = hyperparameterOneValue # Default = 4
                learning_rate = 0.15 # Default = 0.15
                subsample = 0.5 # Default = 0.5

            # Reading Data
            if nJets == 2:
                df_k1_2 = pd.read_csv('../CSV/VHbb_data_2jet_even.csv')
                df_k2_2 = pd.read_csv('../CSV/VHbb_data_2jet_odd.csv')

            else:
                df_k1_2 = pd.read_csv('../CSV/VHbb_data_3jet_even.csv')
                df_k2_2 = pd.read_csv('../CSV/VHbb_data_3jet_odd.csv')

            # Initialising BDTs
            xgb_even = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample)
            xgb_odd = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample)

            # Training using threads
            def train_even():
                xgb_even.fit(df_k1_2[variables], df_k1_2['Class'], sample_weight=df_k1_2['training_weight'])
            def train_odd():
                xgb_odd.fit(df_k2_2[variables], df_k2_2['Class'], sample_weight=df_k2_2['training_weight'])

            t = threading.Thread(target=train_even)
            t2 = threading.Thread(target=train_odd)

            t.start()
            t2.start()
            t.join()
            t2.join()

            # Scoring
            scores_even = xgb_odd.predict_proba(df_k1_2[variables])[:,1]
            scores_odd = xgb_even.predict_proba(df_k2_2[variables])[:,1]

            df_k1_2['decision_value'] = ((scores_even-0.5)*2)
            df_k2_2['decision_value'] = ((scores_odd-0.5)*2)
            df = pd.concat([df_k1_2,df_k2_2])

            # Calculating Sensitivity
            if nJets == 2:
                result_2 = calc_sensitivity_with_error(df)
                print(str(nJets) + " Jet using the Standard BDT: "+ str(result_2[0]) + " ± "+ str(result_2[1]))

            else:
                result_3 = calc_sensitivity_with_error(df)
                print(str(nJets) + " Jet using the Standard BDT: "+ str(result_3[0]) + " ± "+ str(result_3[1]))

        result_combined = totalSensitivity(result_2[0],result_3[0],result_2[1],result_3[1])

        print("Combined Sensitivity", result_combined[0], "±",result_combined[1])
        print("Time Taken", time.time() - start)

        dataset[i,0] = hyperparameterOneValue
        dataset[i,1] = hyperparameterTwoValue
        dataset[i,2] = result_2[0]
        dataset[i,3] = result_2[1]
        dataset[i,4] = result_3[0]
        dataset[i,5] = result_3[1]
        dataset[i,6] = result_combined[0] #combined
        dataset[i,7] = result_combined[1] #combined Uncertainty
        dataset[i,8] = time.time() - start

        i = i + 1

dfDataset = pd.DataFrame(dataset)
dfDataset.to_csv("XGBoost_MaxDepth_vs_NEstimators.csv")

def convertForMatrix(dataset, xVariableLength, yVariableLength):

    convertedDataset = np.zeros((yVariableLength, xVariableLength))

    k = 0

    for i in range(0, yVariableLength):
        for j in range(0, xVariableLength):
            convertedDataset[i, j] = dataset[j + k]

        k = k + xVariableLength

    return convertedDataset

#### Plot Sensitivity Grid ####
graphs = ['2 Jets', '3 Jets', 'Combined']

for i in graphs:

    if i == '2 Jets':
        data = convertForMatrix(dataset[:,2], len(hyperparameterOne), len(hyperparameterTwo))

    if i == '3 Jets':
        data = convertForMatrix(dataset[:,4], len(hyperparameterOne), len(hyperparameterTwo))

    if i == 'Combined':
        data = convertForMatrix(dataset[:,6], len(hyperparameterOne), len(hyperparameterTwo))

    # Draw a heatmap with the numeric values in each cell
    plt.figure(figsize=(20,8))
    ax = plt.axes()

    index = hyperparameterTwo[::-1]
    cols = hyperparameterOne

    df = pd.DataFrame(data, index=index, columns=cols)
    sns.heatmap(data,  cmap="RdYlGn", annot=True, yticklabels=index, xticklabels=cols,  annot_kws={"size":20}, fmt='.3f', cbar=True, cbar_kws={'label': 'Sensitivity'})

    ax.tick_params(axis='y', labelsize=20, rotation=0)
    ax.tick_params(axis='x', labelsize=20, rotation=0)

    ax.set_xlabel(hyperparameterOneName, size=20)
    ax.set_ylabel(hyperparameterTwoName, size=20)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    ax.figure.axes[-1].yaxis.label.set_size(20)

    figureName = "XGBoost_" + str(hyperparameterOneName + "_vs_" + str(hyperparameterTwoName) + ".pdf"
    fig = plt.gcf()
    plt.savefig(figureName, dpi=100, bbox_inches='tight')
    plt.show()
