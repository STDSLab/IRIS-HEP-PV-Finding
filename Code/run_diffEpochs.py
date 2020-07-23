from PV_Experiments import *

print('loading events...')
labelled = loadAndPrepareAllEvents('data/', 100, 10)
print('finished loading events')

print('processing data...')
allData = genDataForEvents(labelled, K=1, featuresList=None, altFeaturesSize=3, altFeaturesData=1, adjMatrixMode='zip',
                           adjFilterKernel='gaussian', delta=5.4, randEdgeProbability=0.5)
print('finished processing data')

paramVals = [1,2,3,4,10,25,50]
learningRate = 0.1
fileName = 'res/newEpochTest' #Experiment results are saved here

print('Running experiment...')
res = exp_SGC_diff_Epoch(allData,paramVals,learningRate,fileName,kFolds=5,shouldShuffleY=False)
print('finished experiment')