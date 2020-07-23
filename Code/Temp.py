from PV_methods import *
from PV_Experiments import *
import numpy as np
import itertools

print('loading events...')
labelled = loadAndPrepareAllEvents('../Data/', 100, 1)
print('finished loading events')

allData = genDataForDiffK(labelled, [1,2,3,4], featuresList=None, altFeaturesSize=3, altFeaturesData=1, adjMatrixMode='ones',
                           adjFilterKernel='gaussian', delta=5.4, randEdgeProbability=0.5)
