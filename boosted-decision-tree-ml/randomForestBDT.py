# Random Forest Classifier
# Author: Louis Heery

import pandas
import numpy
import sys
sys.path.append("../")
sys.path.append("../dataset-and-plotting")
import pickle

import matplotlib.cm as cm
from sklearn.preprocessing import scale

from bdtPlotting import *
from sensitivity import *
from xgboost import XGBClassifier
from IPython.display import display
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import threading


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

start = time.time()

for nJets in [2,3]:

    print("************")
    print("STARTED RandomForest Classifier")

    # Defining BDT Parameters and Input Variables
    if nJets == 2:

        variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]
        n_estimators = 200
        max_depth = 4
        learningRate = 0.15

    else:

        variables_3 = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont','nTrackJetsOR',]

        n_estimators = 300
        max_depth = 4
        learningRate = 0.15

    # Reading Data
    if nJets == 2:
        dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
        dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

    else:
        dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv')
        dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv')


    # Initialising Classifier
    bdtEven = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=0.01)
    bdtOdd = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=0.01)

    # Training BDT
    print("Training the " + str(nJets) + " Jet Dataset")

    bdtEven.fit(dfEven[variables], dfEven['Class'], sample_weight=dfEven['training_weight'])
    bdtOdd.fit(dfOdd[variables], dfOdd['Class'], sample_weight=dfOdd['training_weight'])

    # Calculate Decision Value
    dfEven['decision_value'] = bdtOdd.predict_proba(dfEven[variables])[:,1]
    dfOdd['decision_value'] = bdtEven.predict_proba(dfOdd[variables])[:,1]

    df = pd.concat([dfEven,dfOdd])
    figureName = "RandomForest_" + str(nJets) + "Jets_" + str(n_estimators) + "estimators_" + str(max_depth) + "depth_" + str(learningRate) + "learnrate.pdf"

    h1, ax = final_decision_plot(df, figureName)

    # Calculating Sensitivity
    if nJets == 2:
        sensitivity2Jet = calc_sensitivity_with_error(df)
        print(str(nJets) + " Jet using the Standard BDT: "+ str(sensitivity2Jet[0]) + " ± "+ str(sensitivity2Jet[1]))

    else:
        sensitivity3Jet = calc_sensitivity_with_error(df)
        print(str(nJets) + " Jet using the Standard BDT: "+ str(sensitivity3Jet[0]) + " ± "+ str(sensitivity3Jet[1]))

sensitivityCombined = totalSensitivity(sensitivity2Jet[0],sensitivity3Jet[0],sensitivity2Jet[1],sensitivity3Jet[1])

print("Combined Sensitivity", sensitivityCombined[0], "±",sensitivityCombined[1])
print("Time Taken", time.time() - start)
print("FINISHED")
print("************")
