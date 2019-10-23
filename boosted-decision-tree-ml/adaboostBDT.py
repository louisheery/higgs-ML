# AdaBoost Boosted Decision Tree Classifier
# Author: Louis Heery

import sys

import numpy as np
import pandas as pd

sys.path.append("../")
sys.path.append("../dataset-and-plotting")

from bdtPlotting import *
from sensitivity import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time


def totalSensitivity(A, B, errorA, errorB):
    totalSensitivity = np.sqrt(A ** 2 + B ** 2)
    totalError = np.sqrt(
        ((A * errorA) / np.sqrt(A ** 2 + B ** 2)) ** 2 + ((B * errorB) / np.sqrt(A ** 2 + B ** 2)) ** 2)

    return (totalSensitivity, totalError)


start = time.time()

for nJets in [2, 3]:

    print("************")
    print("STARTED AdaBoost Classifier")

    # Defining BDT Parameters and Input Variables
    if nJets == 2:
        variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV',
                     'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR']
        n_estimators = 50
        max_depth = 4
        learning_rate = 0.15

    else:
        variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ',
                     'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont', 'nTrackJetsOR']
        n_estimators = 50
        max_depth = 4
        learning_rate = 0.15

    # Reading Data
    if nJets == 2:
        dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
        dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

    else:
        dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv')
        dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv')

    # Initialising Classifier
    bdtEven = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=0.01),
                                 learning_rate=learning_rate, algorithm="SAMME", n_estimators=n_estimators)

    bdtOdd = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=0.01), learning_rate=0.15,
                                algorithm="SAMME", n_estimators=n_estimators)

    # Training BDT
    print("Training the " + str(nJets) + " Jet Dataset")
    bdtEven.fit(dfEven[variables], dfEven['Class'], sample_weight=dfEven['training_weight'])
    bdtOdd.fit(dfOdd[variables], dfOdd['Class'], sample_weight=dfOdd['training_weight'])

    # Scoring of BDT
    dfEven['decision_value'] = bdtOdd.decision_function(dfEven[variables]).tolist()
    dfOdd['decision_value'] = bdtEven.decision_function(dfOdd[variables]).tolist()

    df = pd.concat([dfEven, dfOdd])

    figureName = "AdaBoost_" + str(nJets) + "Jets_" + str(n_estimators) + "estimators_" + str(
        max_depth) + "depth_" + str(learning_rate) + "learnrate.pdf"

    h1, ax = final_decision_plot(df, figureName)

    # Calculating Sensitivity
    if nJets == 2:
        sensitivity2Jet = calc_sensitivity_with_error(df)
        print(str(nJets) + " Jet using the Standard BDT: " + str(sensitivity2Jet[0]) + " ± " + str(sensitivity2Jet[1]))

    else:
        sensitivity3Jet = calc_sensitivity_with_error(df)
        print(str(nJets) + " Jet using the Standard BDT: " + str(sensitivity3Jet[0]) + " ± " + str(sensitivity3Jet[1]))

sensitivityCombined = totalSensitivity(sensitivity2Jet[0], sensitivity3Jet[0], sensitivity2Jet[1], sensitivity3Jet[1])

print("Combined Sensitivity = ", sensitivityCombined[0], "±", sensitivityCombined[1])
print("Time Taken = ", time.time() - start)
print("FINISHED")
print("************")
