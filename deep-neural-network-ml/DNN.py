# Deep Neural Network Classifier
# Author: Louis Heery

import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
from keras.layers import Input, Dense, Dropout, Flatten
from keras import backend as K
import time

import sys
sys.path.append("../dataset-and-plotting")
from nnPlotting import *

def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

print("STARTED")
start = time.time()

for nJets in [2,3]:

    print("STARTED TRAINING " + str(nJets) + " Jet Neural Network")

    if nJets == 2:
        variables = ['dRBB','mBB','pTB1', 'pTB2', 'MET','dPhiVBB','dPhiLBmin','Mtop','dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTags', 'nTrackJetsOR']

    else:
        variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont', 'nTags', 'nTrackJetsOR']

    # Read in Data
    if nJets == 2:
        dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
        dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

    else:
        dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv')
        dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv')

    # Process Even Events
    xEven = scale(dfEven[variables].to_numpy())
    yEven = dfEven['Class'].to_numpy()
    wEven = dfEven['training_weight'].to_numpy()

    # Process Odd Events
    xOdd = scale(dfOdd[variables].to_numpy())
    yOdd = dfOdd['Class'].to_numpy()
    wOdd = dfOdd['training_weight'].to_numpy()

    # Define Architecture
    def DNNClassifier():

        model = Sequential()

        # Add Layers
        model.add(Dense(units=14, input_shape=(xEven.shape[1],), activation='relu')) # 1st layer
        model.add(Dense(14, init='uniform', activation='relu')) # hidden layer
        model.add(Dense(14, init='uniform', activation='relu')) # hidden layer
        model.add(Dense(1,activation='sigmoid')) # output layer
        model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
        return model

    # Create and Compile models
    modelEven = DNNClassifier()
    modelOdd = DNNClassifier()

    # Set these parameters
    epochs = 200
    batchSize = 100

    # Train Model
    modelEven.fit(xEven,yEven, sample_weight = wEven, epochs=epochs, batch_size=batchSize, verbose = 1)
    modelOdd.fit(xOdd,yOdd, sample_weight = wOdd, epochs=epochs, batch_size=batchSize, verbose = 1)

    print("Time Taken = " + str(round(time.time()-start,2))+"s")

    ## EVALUATION DNN & Plots
    dfOdd['decision_value'] = modelEven.predict_proba(xOdd)
    dfEven['decision_value'] = modelOdd.predict_proba(xEven)
    df = pd.concat([dfOdd,dfEven])

    figureName = "DNN_" + str(nJets) + "Jets"+ ".pdf"
    nn_output_plot(df, figureName)

    if nJets == 2:
        df.to_csv("DNNOPTIMALALL_2jet_200epoch.csv")
        sensitivity2Jet = sensitivity_NN(df)

    else:
        df.to_csv("DNNOPTIMALALL_3jet_200epoch.csv")
        sensitivity3Jet = sensitivity_NN(df)

    print("FINISHED TRAINING " + str(nJets) + " Jet Neural Network")

print("2 Jet Sensitivity: " + str(sensitivity2Jet[0]) + " ± "+ str(sensitivity2Jet[1]))
print("3 Jet Sensitivity: "+ str(sensitivity3Jet[0]) + " ± "+ str(sensitivity3Jet[1]))

sensitivityCombined = totalSensitivity(sensitivity2Jet[0],sensitivity3Jet[0],sensitivity2Jet[1],sensitivity3Jet[1])

print("Combined Sensitivity", sensitivityCombined[0], "±",sensitivityCombined[1])
print("Total Time Taken", time.time() - start)
print("FINISHED")
