# Author: Louis Heery
import pandas
import numpy
import sys
sys.path.append("../")
sys.path.append("../plotting/")
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from ../bdtPlotting import *
from ../sensitivity import *
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
    totalSensitivitB = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivitB,totalError)

variables_2_all = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]
variables_3_all = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']

dataset = np.zeros((len(variables_3_all), 8))

for i in range(0, len(variables_3_all)):
    print ("Variable Removed: " + variables_3_all[i])

    if i <= len(variables_2_all):
        variables_2 = variables_2_all[:i] + variables_2_all[i+1 :]

    variables_3 = variables_3_all[:i] + variables_3_all[i+1 :]

    start = time.time()

    for nJets in [2,3]:
        print("STARTED " + str(nJets) + " Jets with " + str(i) + " Variable Removed")
        if nJets == 2:
            variables = variables_2
            n_estimators = 200
            max_depth = 14
            learning_rate = 0.15
            subsample = 0.5

        else:
            variables = variables_3
            n_estimators = 200
            max_depth = 14
            learning_rate = 0.15
            subsample = 0.5

        # Reading Data
        if nJets == 2:
            df_k1_2 = pd.read_csv('../CSV/VHbb_data_2jet_even.csv')
            df_k2_2 = pd.read_csv('../CSV/VHbb_data_2jet_odd.csv')

        else:
            df_k1_2 = pd.read_csv('../CSV/VHbb_data_3jet_even.csv')
            df_k2_2 = pd.read_csv('../CSV/VHbb_data_3jet_odd.csv')

        xgb_even = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample)
        xgb_odd = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample)

        xgb_even.fit(df_k1_2[variables], df_k1_2['Class'], sample_weight=df_k1_2['training_weight'])
        xgb_odd.fit(df_k2_2[variables], df_k2_2['Class'], sample_weight=df_k2_2['training_weight'])

        scores_even = xgb_odd.predict_proba(df_k1_2[variables])[:,1]
        scores_odd = xgb_even.predict_proba(df_k2_2[variables])[:,1]

        df_k1_2['decision_value'] = ((scores_even-0.5)*2)
        df_k2_2['decision_value'] = ((scores_odd-0.5)*2)
        df = pd.concat([df_k1_2,df_k2_2])

        #figureName = "VariationOfOptimalSensitivity_DecisionPlot_" + str(nJets) + "Jet" + ".pdf"
        #h1,ax = final_decision_plot(df, figureName)

        if nJets == 2:
            result_2 = calc_sensitivity_with_error(df)
            dataset[i,0] = result_2[0]
            dataset[i,1] = result_2[1]
            print(str(nJets) + " Jet using the Standard BDT: "+ str(result_2[0]) + " ± "+ str(result_2[1]))

        else:
            result_3 = calc_sensitivity_with_error(df)
            dataset[i,2] = result_3[0]
            dataset[i,3] = result_3[1]
            print(str(nJets) + " Jet using the Standard BDT: "+ str(result_3[0]) + " ± "+ str(result_3[1]))

    result_combined = totalSensitivity(result_2[0],result_3[0],result_2[1],result_3[1])
    dataset[i,4] = result_combined[0] # combined
    dataset[i,5] = result_combined[1] # combined Uncertainty
    dataset[i,6] = time.time() - start # time taken

    print("Combined Sensitivity", result_combined[0], "±",result_combined[1])
    print("Time Taken", time.time() - start)


####### Graphs #######
graphs = ['2 Jets', '3 Jets', 'Combined']

for i in graphs:
    if i == '2 Jets':
        df = pd.DataFrame(dataset[0:len(variables_2_all),0:2])
        df["variable_removed"] = variables_2_all

    if i == '3 Jets':
        df = pd.DataFrame(dataset[:,2:4])
        df["variable_removed"] = variables_3_all

    if i == 'Combined':
        df = pd.DataFrame(dataset[:,4:6])
        df["variable_removed"] = variables_3_all

    df_sorted = df.sort_values(by=[0])

    if i == '2 Jets':
        x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        #print(df_sorted[0])

    if i == '3 Jets':
        x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    if i == 'Combined':
        x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    y = df_sorted[0]
    yerr = df_sorted[1]

    plt.figure()

    if i == '2 Jets':
        plt.xticks(x, df_sorted['variable_removed'], rotation=90)

    if i == '3 Jets':
        plt.xticks(x, df_sorted['variable_removed'], rotation=90)

    if i == 'Combined':
        plt.xticks(x, df_sorted['variable_removed'], rotation=90)

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
