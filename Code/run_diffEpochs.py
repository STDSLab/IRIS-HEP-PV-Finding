from SGC_diffHypParamExperiments import *
import sys

# The below was never tested, but is supposed to take arguments from the command line and run the exp.
# Restore these if script doesnt work

# This script is supposed to run the different epochs experiment

# Author: Alan Joshua Aneeth Jegaraj

# @params
# fileName, paramVals, eventFile,shouldShuffleY, learningRate=0.01, subEvents
args = sys.argv
fileName = args[0]
paramVals = args[1]
eventFile = args[2]
shouldShuffleY = args[3]
if len(args) == 5:
    learningRate = 0.01
else:
    learningRate = args[4]

print('loading events...')
labelled = loadAndPrepareAllEvents('data/', eventFile)
print('finished loading events')

print('processing data...')
allData = genDataForEvents(labelled, K=1, featuresList=None, altFeaturesSize=3, altFeaturesData=1,
                           adjMatrixMode='zip',
                           adjFilterKernel='gaussian', delta=5.4, randEdgeProbability=0.5)
print('finished processing data')

print('Running experiment...')
res = exp_SGC_diff_Epoch(allData, paramVals, learningRate, fileName, kFolds=5, shouldShuffleY=False)
print('finished experiment')

# print('loading events...')
# labelled = loadAndPrepareAllEvents('data/', 100, 10)
# print('finished loading events')
#
# print('processing data...')
# allData = genDataForEvents(labelled, K=1, featuresList=None, altFeaturesSize=3, altFeaturesData=1, adjMatrixMode='zip',
#                            adjFilterKernel='gaussian', delta=5.4, randEdgeProbability=0.5)
# print('finished processing data')
#
# paramVals = [1,2,3,4,10,25,50]
# learningRate = 0.01
# fileName = 'res/newEpochTest' #Experiment results are saved here
#
# print('Running experiment...')
# res = exp_SGC_diff_Epoch(allData,paramVals,learningRate,fileName,kFolds=5,shouldShuffleY=False)
# print('finished experiment')
