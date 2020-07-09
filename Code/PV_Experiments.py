from PV_methods import *
from sklearn.model_selection import KFold
import numpy as np
import pickle

# This function is used to train and test a model on different total epoch values
# A new model is initialized per total epoch value, and is trained for that many number of epochs.
# Cross validation is then performed, and the loss and accuracy are recorded

def exp_SGC_diff_Epoch(allData, paramVals, learningRate, savedModelsDir, k_FoldSplits=5, shouldShuffleY=False):
    if shouldShuffleY:
        print('shuffling y labels...')
        for key in allData.keys():
            np.random.shuffle(allData[key]['y'])
        print('finished shuffling')

    with open(savedModelsDir + f'/ParamVals.pickle', 'wb') as handle:
        pickle.dump(paramVals, handle, protocol=pickle.HIGHEST_PROTOCOL)

    upperEval = {'loss': [], 'accuracy': []}
    eventIds = np.array(list(allData.keys()))

    model = genModel(allData[0]['X'].shape[1], allData[0]['y'].shape[1], learning_rate=learningRate, shouldUseBias=True)

    for _ in range(paramVals[-1]):

        kf = KFold(n_splits=k_FoldSplits, shuffle=True)
        midTestEval = []

        for train_index, test_index in kf.split(allData):

            train_index = eventIds[train_index]
            test_index = eventIds[test_index]

            train = [allData[key] for key in train_index]
            test = [allData[key] for key in test_index]

            for key in range(len(train)):
                X = train[key]['X']
                y = train[key]['y']
                fltr = train[key]['fltr']
                model.fit([X, fltr],
                          y,
                          epochs=1,
                          batch_size=X.shape[0]
                          )
                print('Epoch num: ', _, ' Event: ', key)

            if _ + 1 in paramVals:
                testEval = []
                for key in range(len(test)):
                    X = test[key]['X']
                    y = test[key]['y']
                    fltr = test[key]['fltr']
                    testEval.append(model.evaluate([X, fltr],
                                                   y=y,
                                                   batch_size=X.shape[0],
                                                   ))
                midTestEval.append(np.average(np.array(testEval), axis=0))

        if _ + 1 in paramVals:
            averageTest = np.average(np.array(midTestEval), axis=0)
            upperEval['loss'].append(averageTest[0])
            upperEval['accuracy'].append(averageTest[1])

            model.save_weights(savedModelsDir + f'/EpochCount:{_ + 1}.h5')

            with open(savedModelsDir + f'/EpochCount:{_ + 1}.pickle', 'wb') as handle:
                pickle.dump(upperEval, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return upperEval


# This function is used to train and test a model on different K values for the same number of training epochs
# A new model is initialized per K value, and is trained for the specified number of epochs.
# Cross validation is then performed, and the loss and accuracy are recorded

def exp_SGC_diff_k(all_labelled, paramVals, learningRate, savedModelsDir, epochs=100, shouldShuffleY=False,
                   k_FoldSplits=5, featuresList=None, altFeaturesSize=3, altFeaturesData=1, adjMatrixMode='zip',
                   adjFilterKernel='gaussian', delta=5.4):
    upperEval = {'loss': [], 'accuracy': []}

    with open(savedModelsDir + f'/Diff_Epoch_ParamVals.pickle', 'wb') as handle:
        pickle.dump(paramVals, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for param in paramVals:

        print(f'processing data for k={param}...')
        allData = genDataForEvents(all_labelled, K=param, featuresList=featuresList, altFeaturesSize=altFeaturesSize,
                                   altFeaturesData=altFeaturesData,
                                   adjMatrixMode=adjMatrixMode, adjFilterKernel=adjFilterKernel, delta=delta)
        # Shuffles y labels
        if shouldShuffleY:
            for key in allData.keys():
                np.random.shuffle(allData[key]['y'])
        print('finished processing data')

        eventIds = np.array(list(allData.keys()))
        model = genModel(allData[0]['X'].shape[1], allData[0]['y'].shape[1], learning_rate=learningRate,
                         shouldUseBias=True)

        print(f'started training for epochs={epochs} for k={param}...')
        for _ in range(epochs):
            kf = KFold(n_splits=k_FoldSplits, shuffle=True)
            for train_index, test_index in kf.split(allData):

                train_index = eventIds[train_index]
                # test_index = eventIds[test_index]

                train = [allData[key] for key in train_index]
                # test = [allData[key] for key in test_index]

                for key in range(len(train)):
                    X = train[key]['X']
                    y = train[key]['y']
                    fltr = train[key]['fltr']
                    model.fit([X, fltr],
                              y,
                              epochs=1,
                              batch_size=X.shape[0]
                              )
                    print('Epoch num: ', _, ' Event: ', key)

        print(f'Finished training for epochs={epochs} for k={param}...')

        print(f'Started cross validation for k={param}...')
        # Perform cross validation after training for 5
        kf = KFold(n_splits=k_FoldSplits, shuffle=True)
        midTestEval = []

        for train_index, test_index in kf.split(allData):
            test_index = eventIds[test_index]
            test = [allData[key] for key in test_index]

            testEval = []
            for key in range(len(test)):
                X = test[key]['X']
                y = test[key]['y']
                fltr = test[key]['fltr']
                testEval.append(model.evaluate([X, fltr],
                                               y=y,
                                               batch_size=X.shape[0],
                                               ))

            midTestEval.append(np.average(np.array(testEval), axis=0))

        averageTest = np.average(np.array(midTestEval), axis=0)
        upperEval['loss'].append(averageTest[0])
        upperEval['accuracy'].append(averageTest[1])

        print(f'Finished cross validation for k={param}')

        print(f'Saving results for k={param}...')

        model.save_weights(savedModelsDir + f'/K:{param}.h5')
        with open(savedModelsDir + f'/K:{param}.pickle', 'wb') as handle:
            pickle.dump(upperEval, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Finished saving results for k={param}')

    return upperEval


# The below methods could be used to double check whether results from the above EXP functions are valid

# Since the EXP functions save both the intermediate results and trained model weights for each parameter value, the
# trained model could be loaded at each parameter value and validated with test data to see whether the results from it
# corresponds to the results saved by the EXP function


# This function is used to reconstruct data from the different K experiment

# @ params
# labelled = data being used to cross validate
# paramVals = The datapoints to be searched for and loaded from the readFrom directory
# readFrom = Array of directories to read the experimental data from
# saveTo = Array of directories to save the reconstructed data to
# numNodeFeatures = number of node features
# numYClasses = number of unique y labels
# shouldYShuffle = Boolean value to toggle whether to scramble the y labels or not

def reconstructResultsFromSavedModels_DIFF_K(labelled, paramVals, readFrom,saveTo,numNodeFeatures,numYClasses,shouldYShuffle=None,learningRate=0.1,featuresList=None,altFeaturesData=1,adjMatrixMode='zip', adjFilterKernel='gaussian', delta=5.4,readFormat='K:',saveFormat='K='):
    if len(readFrom) != len(saveTo):
        raise ValueError('The array size of readFrom and saveTo must be the same')

    numOfModels = len(readFrom)
    upperEvals = [{'loss': [], 'accuracy': []} for _ in range(numOfModels)]

    for param in paramVals:

        models = []
        for count in range(numOfModels):
            models.append(genModel(numNodeFeatures, numYClasses, learning_rate=learningRate, shouldUseBias=True))

        for index,model in enumerate(models):
            model.load_weights(f'{readFrom}/{readFormat}{param}.h5')

        print(f'Generating data for K={param}...')
        dat = genDataForEvents(labelled, K=param, featuresList=featuresList, altFeaturesSize=numNodeFeatures,
                                   altFeaturesData=altFeaturesData,
                                   adjMatrixMode=adjMatrixMode, adjFilterKernel=adjFilterKernel, delta=delta)

        allData = []
        if shouldYShuffle is None:
            allData = [dat for _ in range(numOfModels)]
        else:
            if len(shouldYShuffle) != numOfModels:
                raise ValueError('The array size of shouldYShuffle should be the same as the number of model data being reconstructed')

            for index,model in enumerate(models):
                curr = shouldYShuffle[index]
                temp = dat
                if curr:
                    for key in temp.keys():
                        np.random.shuffle(temp[key]['y'])
                allData.append(temp)

        eventIds = np.array(list(dat.keys()))
        print(f'Finished Generating data for K={param}')

        kf = KFold(n_splits=5, shuffle=True)
        midTestEval = [[] for _ in range(numOfModels)]

        print('started cross validation...')
        for train_index, test_index in kf.split(dat):

            test_index = eventIds[test_index]
            testEval = [[] for _ in range(numOfModels)]

            for index in range(numOfModels):
                test = [allData[index][key] for key in test_index]

                for key in range(len(test)):
                    X = test[key]['X']
                    y = test[key]['y']
                    fltr = test[key]['fltr']
                    testEval[index].append(models[index].evaluate([X, fltr],
                                                                    y=y,
                                                                    batch_size=X.shape[0],
                                                                    ))

                midTestEval[index].append(np.average(np.array(testEval[index]), axis=0))

        print('saving results...')
        for index in range(numOfModels):
            averageTest = np.average(np.array(midTestEval[index]), axis=0)
            upperEvals[index]['loss'].append(averageTest[0])
            upperEvals[index]['accuracy'].append(averageTest[1])

            with open(saveTo[index] + f'/{saveFormat}{param}.pickle', 'wb') as handle:
                pickle.dump(upperEvals[index], handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('finished saving results')


# This function is used to reconstruct data from the different epochs experiment

# @ params
# allData = data being used to cross validate
# paramVals = The datapoints to be searched for and loaded from the readFrom directory
# readFrom = directory to read the experiment data from
# saveTo = Directory to save the reconstructed data
# shouldYShuffle = Boolean value to toggle whether to scramble the y labels or not

def reconstructResultsFromSavedModels_DIFF_Epochs(allData, paramVals, readFrom,saveTo,shouldYShuffle=False,learningRate=0.1,readFormat='EpochCount:',saveFormat='epoch='):

    upperEval = {'loss': [], 'accuracy': []}
    eventIds = np.array(list(allData.keys()))

    if shouldYShuffle:
        for key in allData.keys():
            np.random.shuffle(allData[key]['y'])

    for param in paramVals:

        model = genModel(allData[0]['X'].shape[1], allData[0]['y'].shape[1], learning_rate=learningRate, shouldUseBias=True)
        model.load_weights(readFrom + f'/{readFormat}{param}.h5')

        kf = KFold(n_splits=5, shuffle=True)
        midTestEval = []

        print(f'started cross validation for epoch={param}...')
        for train_index, test_index in kf.split(allData):

            test_index = eventIds[test_index]
            test = [allData[key] for key in test_index]

            testEval = []
            for key in range(len(test)):
                X = test[key]['X']
                y = test[key]['y']
                fltr = test[key]['fltr']
                testEval.append(model.evaluate([X, fltr],
                                               y=y,
                                               batch_size=X.shape[0],
                                               ))
            midTestEval.append(np.average(np.array(testEval), axis=0))

        averageTest = np.average(np.array(midTestEval), axis=0)
        upperEval['loss'].append(averageTest[0])
        upperEval['accuracy'].append(averageTest[1])

        print('saving results...')
        with open(saveTo + f'/{saveFormat}{param}.pickle', 'wb') as handle:
            pickle.dump(upperEval, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('finished saving results')