# Author: Louis Heery

import pandas
import numpy
import sys
sys.path.append("../")
sys.path.append("../plotting/")
import pickle

import matplotlib.cm as cm
from sklearn.preprocessing import scale

from ../bdtPlotting import *
from ../sensitivity import *
from xgboost import XGBClassifier
from IPython.display import display
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import threading


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivitB = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivitB,totalError)

start = time.time()

for nJets in [2,3]:

    print("************")
    print("STARTED RandomForest Classifier")

    # Defining BDT Parameters and Input Variables
    if nJets == 2:

        variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]
        n_estimators = 200
        max_depth = 4
        learning_rate = 0.15

    else:

        variables_3 = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont','nTrackJetsOR',]

        n_estimators = 300
        max_depth = 4
        learning_rate = 0.15

    # Reading Data
    if nJets == 2:
        df_k1_2 = pd.read_csv('../CSV/VHbb_data_2jet_even.csv')
        df_k2_2 = pd.read_csv('../CSV/VHbb_data_2jet_odd.csv')

    else:
        df_k1_2 = pd.read_csv('../CSV/VHbb_data_3jet_even.csv')
        df_k2_2 = pd.read_csv('../CSV/VHbb_data_3jet_odd.csv')


    # Initialising Classifier
    bdt_even = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=0.01)
    bdt_odd = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=0.01)

    # Training
    print("Training the " + str(nJets) + " Jet Dataset")

    bdt_even.fit(df_k1_2[variables], df_k1_2['Class'], sample_weight=df_k1_2['training_weight'])

    bdt_odd.fit(df_k2_2[variables], df_k2_2['Class'], sample_weight=df_k2_2['training_weight'])

    # Scoring

    #scores_even = xgb_odd.predict_proba(df_k1_2[variables])[:,1]
    #scores_odd = xgb_even.predict_proba(df_k2_2[variables])[:,1]

    df_k1_2['decision_value'] = bdt_odd.predict_proba(df_k1_2[variables])[:,1]
    df_k2_2['decision_value'] = bdt_even.predict_proba(df_k2_2[variables])[:,1]

    df = pd.concat([df_k1_2,df_k2_2])
    figureName = "RandomForest_" + str(nJets) + "Jets_" + str(n_estimators) + "estimators_" + str(max_depth) + "depth_" + str(learning_rate) + "learnrate.pdf"

    h1, ax = final_decision_plot(df, figureName)

    # Calculating Sensitivity
    if nJets ==2:
        result_2 = calc_sensitivity_with_error(df)
        print(str(nJets) + " Jet using the Standard BDT: "+ str(result_2[0]) + " ± "+ str(result_2[1]))

    else:
        result_3 = calc_sensitivity_with_error(df)
        h1.set_size_inches(8.5*1.2,7*1.2)
        display(h1)
        print(str(nJets) + " Jet using the Standard BDT: "+ str(result_3[0]) + " ± "+ str(result_3[1]))

final_combined = totalSensitivity(result_2[0],result_3[0],result_2[1],result_3[1])

print("Combined Sensitivity", final_combined[0], "±",final_combined[1])
print("Time Taken", time.time() - start)
print("FINISHED")
print("************")
