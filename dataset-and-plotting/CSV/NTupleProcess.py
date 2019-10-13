import sys
sys.path.append("/Users/ishankhurana/anaconda3/lib/python3.6/")
import os
import numpy as np
import pandas as pd
import pickle
from root_numpy import root2array
from copy import deepcopy
os.popen("source /usr/local/bin/thisroot.sh")
# os.popen("rm *.csv")




sample_map = {
    'qqZvvH125': 1,
    'qqWlvH125': 2,
    'Wbb': 3,
    'Wbc': 4,
    'Wcc': 5,
    'Wbl': 6,
    'Wcl': 7,
    'Wl': 8,
    'Zbb': 9,
    'Zbc': 10,
    'Zcc': 11,
    'Zbl': 12,
    'Zcl': 13,
    'Zl': 14,
    'ttbar_mc': 15,
    'ttbar_mc_b': 16,
    'ttbar_mc_c': 17,
    'stopt': 18,
    'stops': 19,
    'stopWt': 20,
    'WW': 21,
    'ZZ': 22,
    'WZ': 23
}

scale_factor_map = { #UPDATE THESE
    2: {
        'Zl': 0.99,
        'Zcl': 1.00,
        'Zcc': 1.37,
        'Zbl': 1.37,
        'Zbc': 1.37,
        'Zbb': 1.37,
        'Wl': 0.93,
        'Wcl': 0.93,
        'Wcc': 1.22,
        'Wbl': 1.22,
        'Wbc': 1.22,
        'Wbb': 1.22,
        'stopWt': 0.97,
        'stopt': 0.97,
        'stops': 0.97,
        'ttbar': 0.92,
        'WW': 0.99,
        'ZZ': 0.99,
        'WZ': 0.99,
        'qqZvvH125': 1.0,
        'qqWlvH125': 1.0,
        'qqZllH125': 1.0,
        'ggZllH125': 1.0,
        'ggZvvH125': 1.0
    }, 3: {
        'Zl': 1.0,
        'Zcl': 1.0,
        'Zcc': 1.20,
        'Zbl': 1.20,
        'Zbc': 1.20,
        'Zbb': 1.20,
        'Wl': 0.95,
        'Wcl': 1.02,
        'Wcc': 1.27,
        'Wbl': 1.27,
        'Wbc': 1.27,
        'Wbb': 1.27,
        'stopWt': 0.94,
        'stopt': 0.94,
        'stops': 0.94,
        'ttbar': 0.92,

        'WW': 0.91,
        'ZZ': 0.91,
        'WZ': 0.91,
        'qqZvvH125': 1.0,
        'qqWlvH125': 1.0,
        'qqZllH125': 1.0,
        'ggZllH125': 1.0,
        'ggZvvH125': 1.0,
    }
}

process_general_map = {
    'ggZllH125': 'VH',  #added this key
        'ggZvvH125': 'VH',  #and this one
        'qqZvvH125': 'VH',
        'qqWlvH125': 'VH',  #l-lepton channel
        'qqZllH125': 'VH',
        'Wbb': 'V+jets',
        'Wbc': 'V+jets',
        'Wcc': 'V+jets',
        'Wbl': 'V+jets',
        'Wcl': 'V+jets',
        'Wl': 'V+jets',
        'Zbb': 'V+jets',
        'Zbc': 'V+jets',
        'Zcc': 'V+jets',
        'Zbl': 'V+jets',
        'Zcl': 'V+jets',
        'Zl': 'V+jets',
        'ttbar': 'ttbar',
        'stopt': 'stop',
        'stops': 'stop',
        'stopWt': 'stop',
        'WW': 'diboson',
        'ZZ': 'diboson',
        'WZ': 'diboson'
}

def set_training_weights_for_all(df):
    """Takes a list of events and sets their renormalised training weights."""

    sig_weight_SUM = sum(df['Class']*df['EventWeight'])
    back_weight_SUM = sum((1-df['Class'])*df['EventWeight'])

    sig_FREQ = sum(df['Class'])
    back_FREQ = len(df['Class'])-sig_FREQ

    sig_scale = sig_FREQ / sig_weight_SUM
    back_scale = back_FREQ / back_weight_SUM

    df['training_weight'] = df['EventWeight'] * (df['Class']*sig_scale + (1-df['Class'])*back_scale)

    return df

def change_class_labels(cDf):
    """
    Changes sample names in dataFrames
    """
    df = deepcopy(cDf)
    print("start",set(df['sample']))
    for i,j in df.iterrows():

        if j['sample'] == "ttbar":
            df.at[i,'sample'] = "ttbar"
        elif j['sample'] == "ttbar_PwHerwigppEG":
            df.at[i,'sample'] = "ttbar_mc_b"
        elif j['sample'] == "ttbar_aMcAtNloPy8EG":
            df.at[i,'sample'] = "ttbar_mc_c"

    print("processed",set(df['sample']))
    return df



branch_names = ["sample", "EventWeight", "EventNumber", "ChannelNumber", "isOdd", "weight",
                "nJ", "nTags", "nSigJet", "nForwardJet", "mBB",
                "dRBB", "dPhiBB", "dEtaBB", "sumPt", "pTB1", "pTB2", "pTBB",
                "pTBBoverMET", "etaB1", "etaB2", "MET", "MPT", "HT", "METHT", "MV1cB1",
                "MV1cB2", "MV1cJ3", "pTJ3", "etaJ3", "dRB1J3", "dRB2J3", "mBBJ",
                "dPhiVBB", "dPhiMETMPT", "dPhiMETdijet", "mindPhi", "BDT",
                "dPhiLBmin", "Mtop", "dYWH", "dEtaWH",
                "dPhiLMET", "pTL", "etaL", "mTW", "pTV",]

unneeded = ['dEtaBB', 'dPhiBB', 'weight',
                                 'dEtaWH', 'dPhiLMET', 'BDT', 'pTL', 'etaL', "sumPt",
                                 "ChannelNumber", "isOdd", "nSigJet", "nForwardJet",
                                 "pTBB", "pTBBoverMET", "etaB1", "etaB2", "dEtaBB",
                                 "HT", "METHT", "MPT",
                                 "etaJ3", "dRB1J3", "dRB2J3", "dPhiMETMPT", "dPhiMETdijet",
                                 "mindPhi", "dEtaWH", "dPhiLMET",
                                 'mBBJ', 'pTJ3']

new_branches = ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB',
 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH',
 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont',"MV1cB1",
 "MV1cB2", "MV1cJ3"]


branch_names = list(set(branch_names)^set(unneeded)) # Removes unneeded branches from branch names

# Adding new branches and ensuring no duplicates
for branch in new_branches:
    branch_names.append(branch)

print(branch_names)
branch_names = list(set(branch_names))
print(branch_names)

# print(branch_names)
# Read in NTuples.
# Output S&B as pseudo 2D ndarrays (array of tuple rows).
signal_direct = root2array("Direct_Signal.root",
                           treename="Nominal",
                           branches=branch_names)

signal_truth = root2array("ggZllH125_new_Direct.root",
                          treename="Nominal",
                          branches=branch_names)

# bkg = ['background_normal.root','ttbar_PwPy8EG_Direct.root','ttbar_PHPP_Direct.root','ttbar_aMcAtNloPy8EG.root']
bkg = ['background_normal.root']
background = []

for i in bkg:
    background.append(root2array(i,treename="Nominal",branches=branch_names))

dfs = [pd.DataFrame(i, columns=branch_names) for i in background]
for background_df in dfs:
    background_df['Class'] = pd.Series(np.zeros(len(background_df)))




print ("NTuple read-in complete.")

# Configure as DataFrames.
signal_direct_df = pd.DataFrame(signal_direct, columns=branch_names)
signal_direct_df['Class'] = pd.Series(np.ones(len(signal_direct_df)))

signal_truth_df = pd.DataFrame(signal_truth, columns=branch_names)
signal_truth_df['Class'] = pd.Series(np.ones(len(signal_truth_df)))

signal_df = pd.concat([signal_direct_df,signal_truth_df])

# Concatenate S&B dfs.
df = pd.concat([signal_df]+dfs)
# Map sample names to ints in sample_map.
# df['sample'] = df['sample'].map(lambda x: sample_map[x])


# Cutflow.
df = df[df['nTags'] == 2]

# Split into 2 jet and 3 jet trainings.
df_2jet = df[df['nJ'] == 2]
df_3jet = df[df['nJ'] == 3]

# Split these once again by even/odd event number.
df_2jet_even = df_2jet[df_2jet['EventNumber'] % 2 == 0]
df_2jet_odd = df_2jet[df_2jet['EventNumber'] % 2 == 1]
df_3jet_even = df_3jet[df_3jet['EventNumber'] % 2 == 0]
df_3jet_odd = df_3jet[df_3jet['EventNumber'] % 2 == 1]

print("start",set(df_3jet_even['sample']))

dataFrames = {"2jet_even":df_2jet_even,"2jet_odd":df_2jet_odd,"3jet_even":df_3jet_even,"3jet_odd":df_3jet_odd}

# Write to CSV

for key in dataFrames:
    df = deepcopy(dataFrames[key])
    df = change_class_labels(df)
    df = df.reset_index(drop = True)
    sampleList =  df['sample'].tolist()
    eventWeightList = df['EventWeight'].tolist()
    postFitWeights = []
    categoryList = []

    for x in range(0,len(sampleList)):
        postFitWeights.append(eventWeightList[x]*scale_factor_map[2][sampleList[x]])
        categoryList.append(process_general_map[sampleList[x]])

    df['post_fit_weight'] = postFitWeights
    df['category'] = categoryList
    df = set_training_weights_for_all(df)

    df.to_csv(path_or_buf="VHbb_data_"+key+'.csv')
