# Dijet-mass Aware Adversarial Neural Network Classifier: Without mBB Distribution Graphs
# Author: Louis Heery

import numpy as np
import pandas as pd
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import *
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import scale
from keras.utils import plot_model
from keras.callbacks import History
import time
import os
from copy import deepcopy

import sys
sys.path.append("../dataset-and-plotting")
from nnPlotting import *
from sensitivity import *

def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

for nJets in [2,3]:
    if nJets == 2:
        variables = ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV']

    else:
        variables = ['nTrackJetsOR', 'MV1cB1_cont', 'MV1cB2_cont', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont']

    # ["Classifier_Loss", "Adversary_Loss","Classifier_Accuracy", "Adversary_Accuracy","Sensitivity", "Sensitivity Uncertainty","AdversaryLossNumber"]
    dataStorage = np.zeros((500, 7))

    start = time.time()

    # Set parameters
    lam = 400
    np.random.RandomState(21)
    train = 'odd'
    test  = 'even'

    # Initalise variables
    sensitivity2Jet = [0,0]
    sensitivity3Jet = [0,0]

    # Prepare data
    dfTrain = pd.read_csv('../dataset-and-plotting/CSV_withBinnedDijetMassValues/ADV_' + str(nJets) + 'jet_batch_odd.csv', index_col=0)
    dfTrainAdversary = dfTrain.loc[dfTrain['Class'] == 0]
    dfTrainAdversary = dfTrainAdversary.reset_index()
    dfTest = pd.read_csv('../dataset-and-plotting/CSV_withBinnedDijetMassValues/ADV_' + str(nJets) + 'jet_batch_even.csv', index_col=0)

    # Make categorical generator bins
    zTrain = to_categorical(dfTrain['mBB_category'], num_classes=10)
    zTrainAdversary = to_categorical(dfTrainAdversary['mBB_category'], num_classes=10)

    # Classifier Neural Network Architecture
    inputs = Input(shape=(dfTrain[variables].shape[1],))
    classifierLayer = Dense(40, activation="linear")(inputs)
    classifierLayer = Dense(40, activation="tanh")(classifierLayer)
    classifierLayer = Dense(40, activation="tanh")(classifierLayer)
    classifierLayer = Dense(1, activation="sigmoid", name='classifier_output')(classifierLayer)
    classifierNN = Model(input=[inputs], output=[classifierLayer])
    classifierNN.name = 'classifierNN'

    # Adversary Neural Network Architecture
    inputsAdversary = Input(shape=(dfTrainAdversary[variables].shape[1],))
    adversaryLayer = classifierNN(inputsAdversary)
    adversaryLayer = Dense(30, activation="tanh")(adversaryLayer)
    adversaryLayer = Dense(10, activation="softmax", name='adversary_output')(adversaryLayer)
    adversaryNN = Model(input=[inputsAdversary], output=[adversaryLayer])
    adversaryNN.name = 'adversaryNN'

    # Initial Training of Classifier ('D')
    optimiserClassifier = SGD(lr=0.001, momentum=0.5, decay=0.00001)
    classifierNN.trainable = True; adversaryNN.trainable = False
    classifierNN.compile(loss='binary_crossentropy', optimizer=optimiserClassifier, metrics=['binary_accuracy'])
    classifierNN.fit(dfTrain[variables], dfTrain['Class'], sample_weight=dfTrain['training_weight'], epochs=1, batch_size=32)


    # Function which tests performance of classifier
    def ClassifierTester(classifierNN,df,lam,stage,test,row,nJets):
        predictionScores = classifierNN.predict(df[variables])[:,0]
        df['decision_value'] = ((predictionScores-0.5)*2)

        print ("Iteration Number ROW = " + str(row))
        figureName = "mBB_ANN_" + str(nJets) + "Jets"+ ".pdf"
        nn_output_plot(df, figureName)

        # Calculate Sensitivity & Save to Dataframe
        sensitivity = sensitivity_NN(df)
        dataStorage[row,4] = sensitivity[0]
        dataStorage[row,5] = sensitivity[1]

        if nJets == 2:
            sensitivity2Jet = sensitivity_NN(df)
        else:
            sensitivity3Jet = sensitivity_NN(df)

        #### GRAPHS ####
        dfSorted = df.sort_values('decision_value')

        ##### Signal Events in LOW (-1.0 to +0.8) & HIGH (+0.8 to +1.0) NN REGION GRAPHS #####
        dfSorted_minus1plus08 = dfSorted.loc[dfSorted['decision_value'] <= 0.8]
        dfSorted_plus08plus1 = dfSorted.loc[dfSorted['decision_value'] > 0.8]

        dfNEW = dfSorted_minus1plus08
        savedSignalData = (dfNEW.loc[dfNEW['Class'] == 1])['mBB_raw'] #you can also use dfNEW['column_name']
        plt.hist([savedSignalData], 400,  label="(-1.0 to +0.8) of $NN_{output}$", stacked=True, alpha=0.75)

        dfNEW2 = dfSorted_plus08plus1
        savedSignalData = (dfNEW2.loc[dfNEW2['Class'] == 1])['mBB_raw'] #you can also use dfNEW['column_name']
        plt.hist([savedSignalData], 50,  label="(+0.8 to +1.0) of $NN_{output}$", stacked=True, alpha=0.75)

        plt.xlabel("mBB, MeV")
        plt.xlim(0,250000)
        plt.xticks(rotation=45)
        plt.ylabel('Events')
        plt.legend()
        plt.title('mBB Signal - in Low & High NN_output Regions')
        plt.grid(True)

        # save figure
        figureName = "mBB_ANN_SignalLowVsHighRegion_" + str(nJets) + "Jet_.pdf"
        fig = plt.gcf()
        plt.savefig(figureName, dpi=100, bbox_inches='tight')
        plt.show()
        fig.clear()
        #####

        print ('Sensitivity = ' + str(sensitivity[0]) + ' +/- ' + str(sensitivity[1]))
        df['sensitivity'] = sensitivity[0]
        df.to_csv(path_or_buf='adv_results/'+test+'_batch_'+stage+'_'+str(lam)+'.csv')

    row = 1
    ClassifierTester(classifierNN,dfTest,lam,'start',test,row,nJets)

    # Initial Training of Adversary ('R')
    DfadversaryNN = Model(input=[inputs], output=[adversaryNN(inputs)])
    optimiserClassOnAdv = SGD(lr=1, momentum=0.5, decay=0.00001)
    classifierNN.trainable = False; adversaryNN.trainable = True
    classOnAdvNN.compile(loss='categorical_crossentropy', optimizer=optimiserClassOnAdv, metrics=['accuracy'])
    classOnAdvNN.fit(dfTrainAdversary[variables], zTrainAdversary, sample_weight=dfTrainAdversary['adversary_weights'],batch_size=128, epochs=1)

    # Construct Model with Combined Loss Function of Adversary & Classifier
    optimiserCombinedSystem = SGD(lr=0.001, momentum=0.5, decay=0.00001)
    combinedNNSystem = Model(input=[inputs], output=[classifierNN(inputs), adversaryNN(inputs)])
    classifierNN.trainable = True; adversaryNN.trainable = False
    combinedNNSystem.compile(loss={'classifierNN':'binary_crossentropy','adversaryNN':'binary_crossentropy'}, optimizer=optimiserCombinedSystem, metrics=['accuracy'], loss_weights={'classifierNN': 1.0,'adversaryNN': -lam})

    # Set Maximum number of Adversary-Classifier Training Loops, and batch size of training
    maxLength = len(dfTrain)
    batch_size = 100

    # Training of Adversary
    for i in range(1,maxLength):

        if (i%1000 == 0):
            print ("Iteration Number: " + str(i))

        shuffledIndices = np.random.permutation(len(dfTrain))[:batch_size]
        modelPerformanceMetrics = combinedNNSystem.train_on_batch(dfTrain[variables].iloc[shuffledIndices], [dfTrain['Class'].iloc[shuffledIndices], zTrain[shuffledIndices]], sample_weight=[dfTrain['training_weight'].iloc[shuffledIndices],dfTrain['adversary_weights'].iloc[shuffledIndices]])

        shuffledIndicesAdverary = np.random.permutation(len(dfTrainAdversary))[:batch_size]
        classOnAdvNN.train_on_batch(dfTrainAdversary[variables].iloc[shuffledIndicesAdverary], zTrainAdversary[shuffledIndicesAdverary], dfTrainAdversary['adversary_weights'].iloc[shuffledIndicesAdverary])

        if (i%1000 == 0):
            print ("Iteration Number: " + str(i))
            row = int(20 + (i/1000) + 2)
            dataStorage[row,0] = modelPerformanceMetrics[1]
            dataStorage[row,1] = modelPerformanceMetrics[2]
            dataStorage[row,2] = modelPerformanceMetrics[3]
            dataStorage[row,3] = modelPerformanceMetrics[4]

            ClassifierTester(classifierNN,dfTest,lam,str(i),test,row,nJets)

    row = int(20 + (maxLength/1000) + 2)
    ClassifierTester(classifierNN,dfTest,lam,"end",test,row,nJets)
    classifierNN.save_weights(str(nJets)+"_"+str(lam)+"_"+train+".h5")

    if nJets == 2:
        print("Final " + str(nJets) + " Jet Sensitivity: " + str(sensitivity2Jet[0]) + " ± "+ str(sensitivity2Jet[1]))

    else:
        print("Final " + str(nJets) + " Jet Sensitivity: " + str(sensitivity3Jet[0]) + " ± "+ str(sensitivity3Jet[1]))


    timeTaken = time.time() - start

    df = pd.DataFrame(dataStorage)
    nameoffile = "mbb_ANN_" + str(nJets) + "jet.csv"
    df.to_csv(nameoffile)


sensitivityCombined = totalSensitivity(sensitivity2Jet[0],sensitivity3Jet[0],sensitivity2Jet[1],sensitivity3Jet[1])
print("Final Combined Sensitivity", sensitivityCombined[0], "±", sensitivityCombined[1])
print("Total Time Taken", time.time() - start)
print("FINISHED")
