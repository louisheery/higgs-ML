# Input Variables Correlation Matrices Plotter
# Author: Louis Heery

import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import pandas as pd
import time

categories = ["VH", "diboson", "ttbar_mc_a", "stop", "V+jets"]

for i in categories:

    for nJets in [2,3]:

        if nJets == 2:
            dfOdd = pd.read_csv('../CSV/VHbb_data_2jet_odd.csv', usecols=(6,8,9,11,12,13,14,15,16,17,20,21,22,23,24,25,26,29))
            dfEven = pd.read_csv('../CSV/VHbb_data_2jet_even.csv', usecols=(6,8,9,11,12,13,14,15,16,17,20,21,22,23,24,25,26,29))

        else:
            dfOdd = pd.read_csv('../CSV/VHbb_data_3jet_odd.csv', usecols=(6,8,9,11,12,13,14,15,16,17,20,21,22,23,24,25,26,29))
            dfEven = pd.read_csv('../CSV/VHbb_data_3jet_even.csv', usecols=(6,8,9,11,12,13,14,15,16,17,20,21,22,23,24,25,26,29))


        dfOddParticularEventType = dfOdd.loc[dfOdd['category'] == i]
        dfEvenParticularEventType = dfEven.loc[dfEven['category'] == i]

        frames = [dfOddParticularEventType, dfEvenParticularEventType]
        dataCombined = pd.concat(frames)

        plt.figure(figsize=(40,40))

        # Plot Correlation Matrix
        ax = plt.axes()
        sns.heatmap(dataCombined.corr(), ax = ax, annot=True,annot_kws={"size":40}, fmt='.1f', cbar=False)
        ax.tick_params(axis='x', labelsize=40, rotation=90)
        ax.tick_params(axis='y', labelsize=40, rotation=0)

        if nJets == 2:
            ax.set_title('2jet - ' + str(i), size=40)

        else:
            ax.set_title('3jet - ' + str(i), size=40)

        figureName = "Correlation_" + str(nJets) + "Jets_" + str(i) + ".pdf"
        fig = plt.gcf()
        plt.savefig(figureName, bbox_inches='tight',dpi=300)  # should before plt.show method
        plt.show()
