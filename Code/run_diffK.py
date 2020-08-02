# The below was never tested, but is supposed to take arguments from the command line and run the exp.
# Restore these if script doesnt work

# This script is supposed to run the different epochs experiment


from SGC_diffHypParamExperiments import *
import sys


# @params
# fileName, paramVals, eventFile,shouldShuffleY, subEvents=None,learningRate=0.01
args = sys.argv
fileName = args[0]
paramVals = args[1]
eventFile = args[2]
shouldShuffleY = args[3]

if len(args) < 5:
    subEvents = 0.01
else:
    subEvents = args[4]

if len(args) < 6:
    learningRate = 0.01
else:
    learningRate = args[5]


print('loading events...')
labelled = loadAndPrepareAllEvents('data/', eventFile)
print('finished loading events')

print('Running experiment...')
res = exp_SGC_diff_K(labelled,paramVals,learningRate,fileName,epochs=10,keyEpochs=[3,4,5],testZero=True,
                     shouldShuffleY=shouldShuffleY,kFolds=5)
print('finished experiment')


# print('loading events...')
# labelled = loadAndPrepareAllEvents('data/', 100, 10)
# print('finished loading events')
#
# paramVals = [1,2,3]
# learningRate = 0.01
# fileName = 'res/newKTest'  #Experiment results are saved here
#
# print('Running experiment...')
# res = exp_SGC_diff_K(labelled,paramVals,learningRate,fileName,epochs=10,keyEpochs=[3,4,5],testZero=True,
#                      shouldShuffleY=False,kFolds=5)

# print('finished experiment')