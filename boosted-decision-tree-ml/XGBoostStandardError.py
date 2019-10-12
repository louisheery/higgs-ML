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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import threading
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
from scipy import stats


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivitB = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivitB,totalError)

start = time.time()

numberOfIterations = 500

dataset = np.zeros((numberOfIterations, 7))

for i in range (0,numberOfIterations):

    print("Training Model " + str(i) + "/" + str(numberOfIterations))

    for nJets in [2,3]:

        # Defining BDT Parameters
        if nJets == 2:
            variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]
            n_estimators = 200 # 150
            max_depth = 4 # 6
            learning_rate = 0.15 # 0.05
            subsample = 0.5 # 0.1

        else:
            variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont','nTrackJetsOR',]
            n_estimators = 200 # 150
            max_depth = 4 # 6
            learning_rate = 0.15 # 0.05
            subsample = 0.5 # 0.1

        # Reading Data
        if nJets == 2:
            df_k1_2 = pd.read_csv('../CSV/VHbb_data_2jet_even.csv')
            df_k2_2 = pd.read_csv('../CSV/VHbb_data_2jet_odd.csv')

        else:
            df_k1_2 = pd.read_csv('../CSV/VHbb_data_3jet_even.csv')
            df_k2_2 = pd.read_csv('../CSV/VHbb_data_3jet_odd.csv')

        # Randomly select 90% of dataset
        df_k1_2_90percent = df_k1_2.sample(frac=0.9)
        df_k2_2_90percent = df_k2_2.sample(frac=0.9)

        # Initialising BDTs
        xgb_even = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,subsample=subsample)
        xgb_odd = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,subsample=subsample)

        # Training using threads
        def train_even():
            xgb_even.fit(df_k1_2_90percent[variables], df_k1_2_90percent['Class'], sample_weight=df_k1_2_90percent['training_weight'])
        def train_odd():
            xgb_odd.fit(df_k2_2_90percent[variables], df_k2_2_90percent['Class'], sample_weight=df_k2_2_90percent['training_weight'])

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
            dataset[i,0] = result_2[0]
            dataset[i,1] = result_2[1]
            print(str(nJets) + " Jet using the Standard BDT: "+ str(result_2[0]) + " ± "+ str(result_2[1]))

        else:
            result_3 = calc_sensitivity_with_error(df)
            dataset[i,2] = result_3[0]
            dataset[i,3] = result_3[1]
            print(str(nJets) + " Jet using the Standard BDT: "+ str(result_3[0]) + " ± "+ str(result_3[1]))

    final_combined = totalSensitivity(result_2[0],result_3[0],result_2[1],result_3[1])
    dataset[i,4] = final_combined[0] #combined
    dataset[i,5] = final_combined[1] #combined Uncertainty
    dataset[i,6] = time.time() - start

    print("Combined Sensitivity", final_combined[0], "±", final_combined[1])

print("Total Time Taken", time.time() - start)


########## Gaussian Graph ##########
graphs = ['2 Jets', '3 Jets', 'Combined']

for i in graphs:

    if i == '2 Jets':
        data = dataset[:,0]

    if i == '3 Jets':
        data = dataset[:,2]

    if i == 'Combined':
        data = dataset[:,4]

    n, bins, patches = plt.hist((data), 50, density=True, alpha=0.7, rwidth=0.75, color='#071BCB')

    # find range for gaussian curve
    xmin, xmax = np.percentile(data, 5), np.percentile(data, 95)
    lnspc = np.linspace(xmin, xmax, len(data))

    m, s = stats.norm.fit(data) # get mean and standard deviation
    pdf_g = stats.norm.pdf(lnspc, m, s) # get theoretical values
    plt.plot(lnspc, pdf_g, label="Norm", color="red") # plot it

    plt.xlabel("Sensitivity")
    plt.xticks(rotation=45)
    plt.ylabel('Events')
    plt.ylim(0,18)
    plt.title(i + "   (" + r'$\mu = $' + str(round(m, 3)) + ", "+ r'$ \sigma = $' + str(round(s,3)) + ")")
    plt.grid(axis='y', alpha=0.75)
    #plt.text((xmin + xmax)/2, 0, r'$\mu = $' + str(round(m, 3)) + '\n' + r'$ \sigma = $' + str(round(s,3)))

    name = i.replace(" ", "_")

    figureName = "XGBoost_500Iterations_" + str(name) + ".pdf"
    fig = plt.gcf()
    plt.savefig(figureName, bbox_inches='tight',dpi=300)
    plt.show()

    print (str(i) + " Mean = ", round(m, 3))
    print (str(i) + " Standard Dev. = ", round(s, 3))
