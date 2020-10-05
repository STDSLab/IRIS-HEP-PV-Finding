from Code.clusteringFunctions import clusterAndEvaluate, cluster_HAC, cluster_FNMTF, cluster_kmeans, cluster_kRuns, \
    cluster_kRuns_Split
from Code.prepareData import loadAndPrepareAllEvents, loadFile
from Code.analyzeResults import compileResultsInTable, compileResultsByPVCombination
import pandas as pd
from Code.plottingUtils import makeTotalPerformancePlot, publishPlot, makePerformanceByPVTypePlot


def runExp(tracks, clusteringFunc, fileName, dir, runCore=False, applyKernel=True, save=True,
           repeatTimes=5, distMetric='euclidean', kernel='cosine', kerParam=5.4, errorRate=1, tolCount=10,
           **kwargs):
    res = clusterAndEvaluate(tracks, clusteringFunc, debug=True,
                             maxIter=1000, tolCount=tolCount, errorRate=errorRate,
                             repeatTimes=repeatTimes,
                             applyKernel=applyKernel,
                             distMetric=distMetric, kernel=kernel, kerParam=kerParam,
                             extraMeta='allTracks', **kwargs)

    if save:
        saveIndEXPResults(res, f'{dir}{fileName}.pickle')  # Saves results to file

    return res


dataDir = '../Data/'
saveDir = '../EXP_results/'
eventFile = 500
saveIndEXPResults = False
saveCompiledResults = False
savePlots = False

kMeans_fileName = ''
HAC_fileName = ''
FNMTF_fileName = ''
kRuns_fileName = ''
kRuns_split_fileName = ''

compiledTotalResults_fileName = 'compRes'
compiledByPVType_fileName = 'resByPVType'

compiledTotalResultsPlot_fileName = ''
compiledByPVTypePlot_fileName = ''

print('loading events...')
tracks = loadAndPrepareAllEvents(dataDir, eventFile).sort_values(by='zip')
print('finished loading events')

# kMeans clustering
kMeans = runExp(tracks, cluster_kmeans, kMeans_fileName, saveDir, repeatTimes=20, save=saveIndEXPResults)

# HAC clustering
HAC = runExp(tracks, cluster_HAC, HAC_fileName, saveDir, repeatTimes=20, save=saveIndEXPResults)

# FNMTF clustering
FNMTF = runExp(tracks, cluster_FNMTF, FNMTF_fileName, saveDir, repeatTimes=20, save=saveIndEXPResults)

# KRuns clustering
KRuns = runExp(tracks, cluster_kRuns, kRuns_fileName, saveDir, repeatTimes=20, save=saveIndEXPResults,
               old_FNMTF_thresh=0.1, thresh=0.8, stopPoint=0, old_FNMTF_rep_count=5)

# KRuns + Split clustering
KRuns_Split = runExp(tracks, cluster_kRuns_Split, kRuns_split_fileName, saveDir, repeatTimes=20, save=saveIndEXPResults,
                     old_FNMTF_thresh=0.1, thresh=0.9, stopPoint=0, old_FNMTF_rep_count=5, diffThresh=25,
                     minTksInCluster=5, tksOutMinThresh=0.5)

compiledResults = compileResultsInTable([kMeans, HAC, FNMTF, KRuns, KRuns_Split],
                                        ['kMeans', 'HAC', 'FNMTF', 'KRuns', 'KRuns_Split'])

# This works only when all 500 events are being clustering, because the keyTable is specific to 500 events.
# If you want to perform this for a different number of event, the key table has to be regenerated for specific events
# The key table just contains labels for all PVs in the selected events based on their characteristics

compiledByPVType = compileResultsByPVCombination([kMeans, HAC, FNMTF, KRuns, KRuns_Split],
                                                 ['kMeans', 'HAC', 'FNMTF', 'KRuns', 'KRuns_Split'],
                                                 keyTable=pd.read_excel(f'{dataDir}keyTable.xlsx'))
if saveCompiledResults:
    compiledResults.to_excel(f'{saveDir}{compiledTotalResults_fileName}')
    compiledByPVType.to_excel(f'{saveDir}{compiledByPVType_fileName}')

fig = makeTotalPerformancePlot(compiledResults)
publishPlot(fig, save=savePlots, dispPlot=True, fileName=compiledTotalResultsPlot_fileName)

fig = makePerformanceByPVTypePlot(compiledByPVType)
publishPlot(fig, save=savePlots, dispPlot=True, fileName=compiledByPVTypePlot_fileName)