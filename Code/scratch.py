# Script used for ideating, basically a scratch file


from SGC_diffHypParamExperiments import *
from FNMTF import FNMTF

print('loading data...')
tracks = loadAndPrepareAllEvents('../Data/', 500, 1)
print('finished loading')

# fig = plt.figure(figsize=(15,10), dpi=100)
# ax = fig.add_subplot(111)
#
# plotDensityPlotForPVZIPDistribution(tracks, ax)
# plotDensityPlotForPVZIPDistribution(tracks[tracks.track_label == -1], ax)
#
# plt.xlim(-50, 200)
# plt.title('PV density distribution (all-tracks[-] vs core-tracks[--])',fontsize='x-large')
# plt.xlabel('ZIP', fontsize='x-large')
# plt.ylabel('Density', fontsize='x-large')
#
# ax.get_legend().remove()
# plt.show()

dir = '../Exp_results/tracksTest/'
plotClusterStackedBarPlots(tracks,0,clusterTracksByHAC,'HAC')
# plotClusterStackedBarPlots(tracks,0,clusterTracksByFNMTF_ZIP,'FNMTF_ZIP')

# res1 = getResultsPerEvent(allTracks[0])
# res2 = getResultsPerEvent(onlyCore[0])
#
# saveResults(res1, f'{dir}allTracks_FNMTF_ZIP_1ev_resPerEv.pickle')
# saveResults(res2, f'{dir}coreTracks_FNMTF_ZIP_1ev_resPerEv.pickle')

# clusteringFunc = clusterTracksByNMF_DOCA
# fileName = '_NMF_DOCA_10'
#
# print('started allTracks...')
# allTracks = clusterAndCalculatePercentageTracksFound(labelled, clusteringFunc, debug=True)
# saveResults(allTracks, f'{dir}allTracks{fileName}.pickle')
#
# print('started only Core...')
# onlyCore = clusterAndCalculatePercentageTracksFound(labelled[labelled.track_label == -1], clusteringFunc, debug=True)
# saveResults(onlyCore, f'{dir}coreTracks{fileName}.pickle')
