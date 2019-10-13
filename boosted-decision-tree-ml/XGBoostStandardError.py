# XGBoost Boosted Decision Tree Classifier: Calculate Standard Error of Model
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
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

start = time.time()

numberOfIterations = 500

dataset = np.zeros((numberOfIterations, 7))

for i in range (0,numberOfIterations):

    print("Training Model " + str(i) + "/" + str(numberOfIterations))

    for nJets in [2,3]:

        # Defining BDT Parameters
        if nJets == 2:
            variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]
            nEstimators = 200 # 150
            maxDepth = 4 # 6
            learningRate = 0.15 # 0.05
            subsample = 0.5 # 0.1

        else:
            variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont','nTrackJetsOR',]
            nEstimators = 200 # 150
            maxDepth = 4 # 6
            learningRate = 0.15 # 0.05
            subsample = 0.5 # 0.1

        # Reading Data
        if nJets == 2:
            dfEven = pd.read_csv('../CSV/VHbb_data_2jet_even.csv')
            dfOdd = pd.read_csv('../CSV/VHbb_data_2jet_odd.csv')

        else:
            dfEven = pd.read_csv('../CSV/VHbb_data_3jet_even.csv')
            dfOdd = pd.read_csv('../CSV/VHbb_data_3jet_odd.csv')

        # Randomly select 90% of dataset
        dfEven90percent = dfEven.sample(frac=0.9)
        dfOdd90percent = dfOdd.sample(frac=0.9)

        # Initialising BDTs
        xgbEven = XGBClassifier(nEstimators=nEstimators,maxDepth=maxDepth,learningRate=learningRate,subsample=subsample)
        xgbOdd = XGBClassifier(nEstimators=nEstimators,maxDepth=maxDepth,learningRate=learningRate,subsample=subsample)

        # Multi-thread BDT Training
        def trainEven():
            xgbEven.fit(dfEven90percent[variables], dfEven90percent['Class'], sample_weight=dfEven90percent['training_weight'])
        def trainOdd():
            xgbOdd.fit(dfOdd90percent[variables], dfOdd90percent['Class'], sample_weight=dfOdd90percent['training_weight'])

        # Specify multiple threaded BDT Training
        t = threading.Thread(target=trainEven)
        t2 = threading.Thread(target=trainOdd)

        t.start()
        t2.start()
        t.join()
        t2.join()

        # Scoring
        scoresEven = xgbOdd.predict_proba(dfEven[variables])[:,1]
        scoresOdd = xgbEven.predict_proba(dfOdd[variables])[:,1]

        dfEven['decision_value'] = ((scoresEven-0.5)*2)
        dfOdd['decision_value'] = ((scores_odd-0.5)*2)
        df = pd.concat([dfEven,dfOdd])

        # Calculating Sensitivity
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

    print("Combined Sensitivity", sensitivityCombined[0], "±", sensitivityCombined[1])

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

    name = i.replace(" ", "_") # Add underscores for file name

    # Save final figure to pdf file
    figureName = "XGBoost_500Iterations_" + str(name) + ".pdf"
    fig = plt.gcf()
    plt.savefig(figureName, bbox_inches='tight',dpi=300)
    plt.show()

    print (str(i) + " Mean = ", round(m, 3))
    print (str(i) + " Standard Dev. = ", round(s, 3))
