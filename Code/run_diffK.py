from PV_Experiments import *

print('loading events...')
labelled = loadAndPrepareAllEvents('data/', 100, 10)
print('finished loading events')

paramVals = [1,2,3]
learningRate = 0.1
fileName = 'res/newKTest'  #Experiment results are saved here

print('Running experiment...')
res = exp_SGC_diff_K(labelled,paramVals,learningRate,fileName,epochs=10,keyEpochs=[3,4,5],testZero=True,
                     shouldShuffleY=False,kFolds=5)
print('finished experiment')