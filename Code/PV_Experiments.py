from PV_methods import *
from sklearn.model_selection import KFold
import numpy as np
import pickle


def saveModelAndResults(model, modelFile, resVals, resFile):
    if model is not None:
        model.save_weights(modelFile)

    with open(resFile, 'wb') as handle:
        pickle.dump(resVals, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadFile(fileName):
    with open(fileName, 'rb') as handle:
        return pickle.load(handle)


def shuffleLabels(data, seed=123456789):
    np.random.seed(seed)
    for key in data.keys():
        np.random.shuffle(data[key]['y'])


# Helper function to test the model and save results to file
def testModel(model, test, fileName):
    finalResults = {}
    testEval = []
    customAcc = []
    customAcc_2 = []
    for key in range(len(test)):
        X = test[key]['X']
        y = test[key]['y']
        fltr = test[key]['fltr']
        testEval.append(model.evaluate([X, fltr],
                                       y=y,
                                       batch_size=X.shape[0],
                                       ))
        pred = model.predict([X, fltr], batch_size=X.shape[0])
        customAcc.append(calcAccuracy(y, pred))
        customAcc_2.append(calcAccuracy_2(y, pred))

    averageTest = np.average(np.array(testEval), axis=0)
    finalResults['loss'].append(averageTest[0])
    finalResults['accuracy'].append(averageTest[1])
    finalResults['custom_accuracy'].append(np.average(np.array(customAcc)))
    finalResults['custom_accuracy_2'].append(np.average(np.array(customAcc_2)))

    saveModelAndResults(model, f'{fileName}.h5', resVals=finalResults,
                        resFile=f'{fileName}.pickle')


# Helper function to train for a given number of total epochs, while saving the results at specified key epochs
def trainTestEpochs(model, train, test, totalEpochs, saveFileDir, filePrefix, k_index, paramVals):

    # Test model before training and save model weights and results
    testModel(model, test, f'{saveFileDir}/{filePrefix}test_FoldInd:{k_index}_ep:0')

    # Train for totalEpochs
    for epochCount in range(totalEpochs):

        # Loop through all events in training set
        for key in range(len(train)):
            X = train[key]['X']
            y = train[key]['y']
            fltr = train[key]['fltr']
            model.fit([X, fltr],
                      y,
                      epochs=1,
                      batch_size=X.shape[0]
                      )
            print('Epoch num: ', epochCount, ' Event: ', key)

        # Test model and save results and model weights only at specified key epochs
        if epochCount + 1 in paramVals:
            testModel(model, test, f'{saveFileDir}/{filePrefix}test_FoldInd:{k_index}_ep:{epochCount + 1}')
            print(f'testing model at epoch {epochCount + 1}')


# This helper function is used to concatenate results from different KFolds
def concatenateAndSaveResults(savedModelsDir, kFolds, paramVals, fileName='FINAL'):
    # Produces and saves final results from intermediate results (Average all K-fold results)
    paramsToLoad = paramVals
    paramsToLoad.insert(0, 0)

    # Loads data from intermediate files into array
    allFoldsData = []
    for k_index in range(kFolds):
        tempFoldResults = {'loss': [], 'accuracy': [], 'custom_accuracy': [], 'custom_accuracy_2': []}
        for epoch in paramsToLoad:
            vals = loadFile(f'{savedModelsDir}/test_kInd:{k_index}_ep:{epoch}.pickle')

            tempFoldResults['loss'].append(vals['loss'])
            tempFoldResults['accuracy'].append(vals['accuracy'])
            tempFoldResults['custom_accuracy'].append(vals['custom_accuracy'])
            tempFoldResults['custom_accuracy_2'].append(vals['custom_accuracy_2'])

        allFoldsData.append(tempFoldResults)

    # Computes the average of results over kFolds, and produces final results
    finalResults = {'loss': [], 'accuracy': [], 'custom_accuracy': [], 'custom_accuracy_2': []}
    for key in finalResults.keys():
        tempArray = []
        for k_index in range(kFolds):
            vals = allFoldsData[k_index][key]
            tempArray.append(vals)
        average = np.average(np.array(tempArray), axis=0)
        finalResults[key] = average

    # Save final results
    saveModelAndResults(None, '', finalResults, f'{savedModelsDir}/{fileName}.pickle')

    return finalResults


# # This function is used to train and test a model on different total epoch values
# # A new model is initialized per total epoch value, and is trained for that many number of epochs.
# # Cross validation is then performed, and the loss and accuracy are recorded

def exp_SGC_diff_Epoch(allData, paramVals, learningRate, savedModelsDir, kFolds=5, KfoldShuff=False,
                         shouldShuffleY=False, seed=123456789):

    # Saves experiment parameters to file
    expParams = {'params': paramVals.insert(0, 0), 'rand_seed': seed,
                 'labels_shuffled': shouldShuffleY, 'learning_rate': learningRate, 'kFolds': kFolds,
                 'modelSize': [allData[0]['X'].shape[1], allData[0]['y'].shape[1]]}

    saveModelAndResults(None, '', resVals=expParams,
                        resFile=f'{savedModelsDir}/expParams.pickle')  # Saved experiment parameters to file

    # Shuffle labels if required
    if shouldShuffleY:
        shuffleLabels(allData, seed)

    eventIds = np.array(list(allData.keys()))

    # Creates K-Fold object to split data into k-folds
    kf = KFold(n_splits=kFolds, shuffle=KfoldShuff)

    # splits train and test data for k-fold cross validation
    for k_index, [train_index, test_index] in enumerate(kf.split(allData)):
        # Gets train and test values
        train_index = eventIds[train_index]
        test_index = eventIds[test_index]
        train = [allData[key] for key in train_index]
        test = [allData[key] for key in test_index]

        # Generates a random SGC model
        model = genModel(train[0]['X'].shape[1], train[0]['y'].shape[1], learning_rate=learningRate,
                         shouldUseBias=True)

        # Trains model, and tests it at the specified key epochs, and saves the test results along with model weights
        # at those key epochs
        trainTestEpochs(model, train, test, paramVals[-1], savedModelsDir,'', k_index, paramVals)
        print(f'Finished the {k_index}th k-fold')

    # Computes final results by averaging results over the different kFolds and saves final results
    res = concatenateAndSaveResults(savedModelsDir, kFolds, paramVals)
    print(f'Finished computing and saving final results')

    return res


# # This function is used to train and test a model on different K values for the same number of training epochs
# # A new model is initialized per K value, and is trained for the specified number of epochs.
# # Cross validation is then performed, and the loss and accuracy are recorded

def exp_SGC_diff_K(data, paramVals, learningRate, savedModelsDir, epochs=100, shouldShuffleY=False,
                   kFolds=5, seed=123456789, KfoldShuff = False, featuresList=None, altFeaturesSize=3, altFeaturesData=1,
                   adjMatrixMode='zip', adjFilterKernel='gaussian', delta=5.4):

    finalResults = []

    # Generates the required data, as each k value requires different data.
    # Larger K values take more time for this step to complete

    print(f'processing data for k={K}...')
    allData = genDataForDiffK(data, paramVals, featuresList=featuresList, altFeaturesSize=altFeaturesSize,
                              altFeaturesData=altFeaturesData,
                              adjMatrixMode=adjMatrixMode, adjFilterKernel=adjFilterKernel, delta=delta)
    eventIds = np.array(list(allData.keys()))

    # Saves experiment parameters to file (if condition to make sure it is saved only once)
    expParams = {'params': paramVals.insert(0, 0), 'rand_seed': seed,
                 'labels_shuffled': shouldShuffleY, 'learning_rate': learningRate, 'kFolds': kFolds,
                 'modelSize': [allData[0]['X'].shape[1], allData[0]['y'].shape[1]]}

    saveModelAndResults(None, '', resVals=expParams,
                        resFile=f'{savedModelsDir}/expParams.pickle')  # Saved experiment parameters to file

    # Shuffle labels accordingly
    if shouldShuffleY:
        shuffleLabels(allData, seed)

    print('finished processing data')

    # Loop through all different K values in paramVals
    for K in paramVals:

        # Creates K-Fold object to split data into k-folds
        kf = KFold(n_splits=kFolds, shuffle=KfoldShuff)

        # splits train and test data for k-fold cross validation
        for k_index, [train_index, test_index] in enumerate(kf.split(allData[K])):

            # Gets train and test values
            train_index = eventIds[train_index]
            test_index = eventIds[test_index]
            train = [allData[key] for key in train_index]
            test = [allData[key] for key in test_index]

            # Generates a random SGC model
            model = genModel(train[0]['X'].shape[1], train[0]['y'].shape[1], learning_rate=learningRate,
                             shouldUseBias=True)

            # Trains model, and tests it at the specified key epochs, and saves the test results along with model
            # weights at those key epochs
            trainTestEpochs(model, train, test, epochs, savedModelsDir, f'K={K}_', k_index, [epochs])

        # Computes results by averaging results over the different kFolds and saves results
        results = concatenateAndSaveResults(savedModelsDir, kFolds, [epochs],fileName=f'FINAL_K={K}')
        finalResults.append(results)
        print(f'Finished computing and saving results for K={K}')

    # Saves final results to file
    saveModelAndResults(None,'',finalResults,f'{savedModelsDir}/ULTIMATE_FINAL.pickle')

    return finalResults
