# XGBoost Boosted Decision Tree Classifier Input Variable Optimisation
# Author: Louis Heery

import pandas
import numpy
import sys
sys.path.append("../")
sys.path.append("../dataset-and-plotting")
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from bdtPlotting import *
from sensitivity import *
from xgboost import XGBClassifier
from IPython.display import display
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import threading
import pandas as pd
import numpy as np
from collections import Counter


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

variables2JetAll = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]
variables3JetAll = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']

dataset = np.zeros((len(variables3JetAll), 8))

for i in range(0, len(variables3JetAll)):
    print ("Variable Removed: " + variables3JetAll[i])

    if i <= len(variables2JetAll):
        variables_2 = variables2JetAll[:i] + variables2JetAll[i+1 :]

    variables_3 = variables3JetAll[:i] + variables3JetAll[i+1 :]

    start = time.time()

    for nJets in [2,3]:
        print("STARTED " + str(nJets) + " Jets with " + str(i) + " Variable Removed")
        if nJets == 2:
            variables = variables_2
            n_estimators = 200
            max_depth = 14
            learningRate = 0.15
            subsample = 0.5

        else:
            variables = variables_3
            n_estimators = 200
            max_depth = 14
            learningRate = 0.15
            subsample = 0.5

        # Reading Data
        if nJets == 2:
            dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
            dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

        else:
            dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv')
            dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv')

        xgbEven = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learningRate=learningRate, subsample=subsample)
        xgbOdd = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learningRate=learningRate, subsample=subsample)

        xgbEven.fit(dfEven[variables], dfEven['Class'], sample_weight=dfEven['training_weight'])
        xgbOdd.fit(dfOdd[variables], dfOdd['Class'], sample_weight=dfOdd['training_weight'])

        scores_even = xgbOdd.predict_proba(dfEven[variables])[:,1]
        scores_odd = xgbEven.predict_proba(dfOdd[variables])[:,1]

        dfEven['decision_value'] = ((scores_even-0.5)*2)
        dfOdd['decision_value'] = ((scores_odd-0.5)*2)
        df = pd.concat([dfEven,dfOdd])


        if nJets == 2:
            sensitivity2Jet = calc_sensitivity_with_error(df)
            dataset[i,0] = sensitivity2Jet[0]
            dataset[i,1] = sensitivity2Jet[1]
            print(str(nJets) + " Jet using the Standard BDT: "+ str(sensitivity2Jet[0]) + " ± "+ str(sensitivity2Jet[1]))

        else:
            sensitivity3Jet = calc_sensitivity_with_error(df)
            dataset[i,2] = sensitivity3Jet[0]
            dataset[i,3] = sensitivity3Jet[1]
            print(str(nJets) + " Jet using the Standard BDT: "+ str(sensitivity3Jet[0]) + " ± "+ str(sensitivity3Jet[1]))

    sensitivityCombined = totalSensitivity(sensitivity2Jet[0],sensitivity3Jet[0],sensitivity2Jet[1],sensitivity3Jet[1])
    dataset[i,4] = sensitivityCombined[0] # combined
    dataset[i,5] = sensitivityCombined[1] # combined Uncertainty
    dataset[i,6] = time.time() - start # time taken

    print("Combined Sensitivity", sensitivityCombined[0], "±",sensitivityCombined[1])
    print("Time Taken", time.time() - start)


####### Graphs #######
graphs = ['2 Jets', '3 Jets', 'Combined']

for i in graphs:
    if i == '2 Jets':
        df = pd.DataFrame(dataset[0:len(variables2JetAll),0:2])
        df["variable_removed"] = variables2JetAll

    if i == '3 Jets':
        df = pd.DataFrame(dataset[:,2:4])
        df["variable_removed"] = variables3JetAll

    if i == 'Combined':
        df = pd.DataFrame(dataset[:,4:6])
        df["variable_removed"] = variables3JetAll

    dfOrdered = df.sort_values(by=[0])

    if i == '2 Jets':
        x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

    if i == '3 Jets':
        x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    if i == 'Combined':
        x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    y = dfOrdered[0]
    yerr = dfOrdered[1]

    plt.figure()

    if i == '2 Jets':
        plt.xticks(x, dfOrdered['variable_removed'], rotation=90)

    if i == '3 Jets':
        plt.xticks(x, dfOrdered['variable_removed'], rotation=90)

    if i == 'Combined':
        plt.xticks(x, dfOrdered['variable_removed'], rotation=90)

    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color='black', label='_nolegend_')
    plt.axhline(y=y.max(), xmin=0, xmax=16, color='blue', label='Baseline Sensitivity')
    plt.title(i)
    plt.legend(loc='lower right')
    plt.ylabel('Sensitivity')
    plt.grid(True, axis='y')

    name = i.replace(" ", "_")

    figureName = "VariationOfOptimalSensitivity_" + str(name) + ".pdf"
    fig = plt.gcf()
    plt.savefig(figureName, bbox_inches='tight',dpi=300)
    plt.show()
