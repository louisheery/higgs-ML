import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import pandas as pd
import time

# ALL_variables = ["dYWH","MV1cJ3_cont","dPhiVBB","pTB2","nTags","mBB","MV1cB1","MV1cB2","pTV","MV1cB1_cont","nTrackJetsOR","pTB1","pTJ3","mBBJ","MV1cB2_cont","mTW","dRBB","MET","Mtop","dPhiLBmin","MV1cJ3"]
# variables = ["dYWH","MV1cJ3_cont","dPhiVBB","pTB2","mBB","MV1cB1","MV1cB2","pTV","MV1cB1_cont","nTrackJetsOR","pTB1","MV1cB2_cont","mTW","dRBB","MET","Mtop","dPhiLBmin","MV1cJ3"]

categories = ["VH", "diboson", "ttbar_mc_a", "stop", "V+jets"]

for i in categories:

    for nJets in [2,3]:

        if nJets == 2:
            data_odd = pd.read_csv('VHbb_data_2jet_odd.csv', usecols=(6,8,9,11,12,13,14,15,16,17,20,21,22,23,24,25,26,29))
            data_even = pd.read_csv('VHbb_data_2jet_even.csv', usecols=(6,8,9,11,12,13,14,15,16,17,20,21,22,23,24,25,26,29))

        else:
            data_odd = pd.read_csv('VHbb_data_3jet_odd.csv', usecols=(6,8,9,11,12,13,14,15,16,17,20,21,22,23,24,25,26,29))
            data_even = pd.read_csv('VHbb_data_3jet_even.csv', usecols=(6,8,9,11,12,13,14,15,16,17,20,21,22,23,24,25,26,29))


        data_odd_certainevent = data_odd.loc[data_odd['category'] == i]
        data_even_certainevent = data_even.loc[data_even['category'] == i]

        frames = [data_odd_certainevent, data_even_certainevent]

        data_combined = pd.concat(frames)

        plt.figure(figsize=(40,40))

        ax = plt.axes()
        sns.heatmap(data_combined.corr(), ax = ax, annot=True,annot_kws={"size":40}, fmt='.1f', cbar=False)
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

        print ('\n\n\n\n')
