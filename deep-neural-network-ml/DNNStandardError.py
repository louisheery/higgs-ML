# Deep Neural Network Classifer: Calculate Standard Error of Model
# Author: Louis Heery

import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
from keras.layers import Input, Dense, Dropout, Flatten
from keras import backend as K
from time import time
from scipy import stats

import sys
sys.path.append("../")
from nnPlotting import *

def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

start = time.time()

numberOfIterations = 2

dataset = np.zeros((numberOfIterations, 7))

for i in range (0,numberOfIterations):

    print("Training Model " + str(i) + "/" + str(numberOfIterations))

    for nJets in [2,3]:

        if nJets == 2:
            variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]

        else:
            variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont','nTrackJetsOR',]

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

        # Define architecture
        def DNNClassifier():
            model = Sequential()
            model.add(Dense(units=14, input_shape=(xEven.shape[1],), activation='relu'))
            model.add(Dense(1,activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
            return model

        # Create and compile models
        modelEven = DNNClassifier()
        modelOdd = DNNClassifier()

        # Set these parameters
        epochs = 200
        batchSize = 100

        # train
        modelEven.fit(xEven,yEven,sample_weight = wEven, epochs=epochs, batch_size=batchSize,verbose = 1)
        modelOdd.fit(xOdd,yOdd,sample_weight = wOdd, epochs=epochs, batch_size=batchSize,verbose = 1)

        ## EVALUATION DNN & Plots
        dfOdd['decision_value'] = modelEven.predict_proba(xOdd)
        dfEven['decision_value'] = modelOdd.predict_proba(xEven)
        df = pd.concat([dfOdd,dfEven])

        # Calculating Sensitivity
        if nJets == 2:
            sensitivity2Jet = sensitivity_NN(df)
            dataset[i,0] = sensitivity2Jet[0]
            dataset[i,1] = sensitivity2Jet[1]
            print(str(nJets) + " Jet using the DNN: "+ str(sensitivity2Jet[0]) + " ± "+ str(sensitivity2Jet[1]))

        else:
            sensitivity3Jet = sensitivity_NN(df)
            dataset[i,2] = sensitivity3Jet[0]
            dataset[i,3] = sensitivity3Jet[1]
            print(str(nJets) + " Jet using the DNN: "+ str(sensitivity3Jet[0]) + " ± "+ str(sensitivity3Jet[1]))

    sensitivityCombined = totalSensitivity(sensitivity2Jet[0],sensitivity3Jet[0],sensitivity2Jet[1],sensitivity3Jet[1])
    dataset[i,4] = sensitivityCombined[0] #combined
    dataset[i,5] = sensitivityCombined[1] #combined Uncertainty
    dataset[i,6] = time.time() - start

    print("Combined Sensitivity", sensitivityCombined[0], "±",sensitivityCombined[1])

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

    name = i.replace(" ", "_")
    figureName = "DNN_500Iterations_" + str(name) + ".pdf"
    fig = plt.gcf()
    plt.savefig(figureName, bbox_inches='tight',dpi=300)
    plt.show()

    print(str(i) + " Mean = ", round(m, 3))
    print(str(i) + " Standard Dev. = ", round(s, 3))
