# Deep Neural Network Classifier: Hyperparameter optimisation script which finds the optimal hyperparameters to train the DNN with.
# Author: Louis Heery

# How to Use:
# 1. [LINE 50] Replace 'hyperparameterOne = ' with the desired Hyperparameter from: maxDepth, nEstimators, learningRate, subSample
# 2. [LINE 51] Replace 'hyperparameterTwo = ' with the desired Hyperparameter from: maxDepth, nEstimators, learningRate, subSample
# 3. [LINE 106, 113, 121, 122] Assign hyperparameterOneValue & hyperparameterTwoValue to their correct hyperparameter, and set the other two hyperparameters to their default value.

import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
from keras.layers import Input, Dense, Dropout, Flatten
from keras import backend as K
from time import time
import seaborn as sns

import sys
sys.path.append("../")
from nnPlotting import *


def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

def convertForMatrix(dataStorage, xVariableLength, yVariableLength):

    convertedDataset = np.zeros((yVariableLength, xVariableLength))

    k = 0

    for i in range(0, yVariableLength):
        for j in range(0, xVariableLength):
            convertedDataset[i, j] = dataStorage[j + k]

        k = k + xVariableLength

    return convertedDataset

epochNumber = [10,20,30,40,50,75,100,200,400,600,800,1000]
batchSize = [50,100,200,500,1000]
optimiserAlgorithm = "SGD" # ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam", optimizers.SGD(lr=0.001, momentum=0.5, decay=0.00001)]
numberOfHiddenLayers = 1 # [1,2,3,4,5,6,7,8,9,10]

hyperparameterOne = epochNumber
hyperparameterTwo = batchSize

hyperparameterOneName = "Epoch Number"
hyperparameterTwoName = "Batch Size"

numberOfIterations = len(hyperparameterOne) * len(hyperparameterTwo)

dataStorage = np.zeros((numberOfIterations, 9))

i = 0

for hyperparameterOneValue in hyperparameterOne:
    for hyperparameterTwoValue in hyperparameterTwo:

        print("Training Model " + str(i + 1) + "/" + str(numberOfIterations) + " with Hyperparameters of")
        print("Hyperparameter One = ", hyperparameterOneValue)
        print("Hyperparameter Two = ", hyperparameterTwoValue)

        start = time.time()

        for nJets in [2,3]:

            if nJets == 2:
                variables = ['dRBB','mBB','pTB1', 'pTB2', 'MET','dPhiVBB','dPhiLBmin','Mtop','dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTags', 'nTrackJetsOR']

            else:
                variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont', 'nTags', 'nTrackJetsOR']

            # Reading Data
            if nJets == 2:
                dfEven = pd.read_csv('../CSV/VHbb_data_2jet_even.csv')
                dfOdd = pd.read_csv('../CSV/VHbb_data_2jet_odd.csv')

            else:
                dfEven = pd.read_csv('../CSV/VHbb_data_3jet_even.csv')
                dfOdd = pd.read_csv('../CSV/VHbb_data_3jet_odd.csv')

            # Even events
            xEven = scale(dfEven[variables].to_numpy())
            yEven = dfEven['Class'].to_numpy()
            wEven = dfEven['training_weight'].to_numpy()

            # Odd events
            xOdd = scale(dfOdd[variables].to_numpy())
            yOdd = dfOdd['Class'].to_numpy()
            wOdd = dfOdd['training_weight'].to_numpy()

            # Build DNN Structure
            def DNNClassifier():
                model = Sequential()

                model.add(Dense(units=14, input_shape=(xEven.shape[1],), activation='relu')) # input layer
                hL = 1 # Total Number of hidden layers

                # add hidden layers
                while hL < numberOfHiddenLayers:
                    model.add(Dense(hiddenLayersSize, init='uniform', activation='relu'))
                    print("1 Extra Hidden Layer Added")
                    hL = hL + 1

                # output layer
                model.add(Dense(1,activation='relu'))
                model.compile(loss='binary_crossentropy', optimizer=optimiserAlgorithm, metrics=['accuracy'])
                return model

            # Create and compile models
            modelEven = DNNClassifier()
            modelOdd = DNNClassifier()

            # Set these parameters
            epochs = hyperparameterOneValue
            batchSize = hyperparameterTwoValue

            # Train NN
            modelEven.fit(xEven,yEven,sample_weight = wEven, epochs=epochs, batch_size=batchSize,verbose = 1)
            modelOdd.fit(xOdd,yOdd,sample_weight = wOdd, epochs=epochs, batch_size=batchSize,verbose = 1)

            ## EVALUATION DNN & Plots
            dfOdd['decision_value'] = modelEven.predict_proba(xOdd)
            dfEven['decision_value'] = modelOdd.predict_proba(xEven)
            df = pd.concat([dfOdd,dfEven])

            if nJets == 2:
                sensitivity2Jet = sensitivity_NN(df)
                print(str(nJets) + " Jet using the DNN: "+ str(sensitivity2Jet[0]) + " ± "+ str(sensitivity2Jet[1]))

            else:
                sensitivity3Jet = sensitivity_NN(df)
                print(str(nJets) + " Jet using the DNN: "+ str(sensitivity3Jet[0]) + " ± "+ str(sensitivity3Jet[1]))

        sensitivityCombined = totalSensitivity(sensitivity2Jet[0],sensitivity3Jet[0],sensitivity2Jet[1],sensitivity3Jet[1])

        print("Combined Sensitivity", sensitivityCombined[0], "±",sensitivityCombined[1])
        print("TOTAL Time Taken", time.time() - start)

        dataStorage[i,0] = hyperparameterOneValue
        dataStorage[i,1] = hyperparameterTwoValue
        dataStorage[i,2] = sensitivity2Jet[0]
        dataStorage[i,3] = sensitivity2Jet[1]
        dataStorage[i,4] = sensitivity3Jet[0]
        dataStorage[i,5] = sensitivity3Jet[1]
        dataStorage[i,6] = sensitivityCombined[0] #combined
        dataStorage[i,7] = sensitivityCombined[1] #combined Uncertainty
        dataStorage[i,8] = time.time() - start

        i = i + 1

dfDataset = pd.DataFrame(dataStorage)
filename = "DNN_" + str(hyperparameterOneName) + "_vs_" + str(hyperparameterTwoName) + ".csv"
dfDataset.to_csv(filename)



#### Plot Sensitivity Grid ####
graphs = ['2 Jets', '3 Jets', 'Combined']

for i in graphs:

    if i == '2 Jets':
        data = convertForMatrix(dataStorage[:,2], len(hyperparameterOne), len(hyperparameterTwo))

    if i == '3 Jets':
        data = convertForMatrix(dataStorage[:,4], len(hyperparameterOne), len(hyperparameterTwo))

    if i == 'Combined':
        data = convertForMatrix(dataStorage[:,6], len(hyperparameterOne), len(hyperparameterTwo))

    # Draw a heatmap with the numeric values in each cell
    plt.figure(figsize=(20,8))
    ax = plt.axes()

    index = hyperparameterTwo[::-1]
    cols = hyperparameterOne

    df = pd.DataFrame(data, index=index, columns=cols)
    sns.heatmap(data,  cmap="RdYlGn", annot=True, yticklabels=index, xticklabels=cols,  annot_kws={"size":20}, fmt='.3f', cbar=True, cbar_kws={'label': 'Sensitivity'})

    ax.tick_params(axis='y', labelsize=20, rotation=0)
    ax.tick_params(axis='x', labelsize=20, rotation=0)

    ax.set_xlabel(hyperparameterOneName, size=20)
    ax.set_ylabel(hyperparameterTwoName, size=20)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    ax.figure.axes[-1].yaxis.label.set_size(20)

    figureName = "DNN_" + str(hyperparameterOneName) + "_vs_" + str(hyperparameterTwoName) + ".pdf"
    fig = plt.gcf()
    plt.savefig(figureName, dpi=100, bbox_inches='tight')
    plt.show()
