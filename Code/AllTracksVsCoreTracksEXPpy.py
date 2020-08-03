# This script runs different clustering algorithms on the loaded events, with running them on both all-tracks and
# only-core

# The results from the experiments are saved in the specified location

# Author: Alan Joshua Aneeth Jegaraj


from PV_methods import *


def runExp(tracks, clusteringFunc, fileName, dir):
    print('started allTracks...')
    allTracks = clusterAndCalculatePercentageTracksFound(tracks, clusteringFunc, debug=True)
    saveResults(allTracks, f'{dir}allTracks{fileName}.pickle')  # Saves results to file

    print('started only Core...')
    onlyCore = clusterAndCalculatePercentageTracksFound(tracks[tracks.track_label == -1], clusteringFunc, debug=True)
    saveResults(onlyCore, f'{dir}coreTracks{fileName}.pickle')


dataDir = 'Data/'
saveDir = 'res/tracksCluster/'
eventFile = 500

print('loading events...')
tracks = loadAndPrepareAllEvents(dataDir, eventFile)  # Loads events, and labels tracks
print('finished loading events')


runExp(tracks, clusterTracksByHAC, '_HAC', saveDir)
runExp(tracks, clusterTracksByHAC_2, '_HAC_5Feat', saveDir)
runExp(tracks, clusterTracksByKMeans, '_kMeans', saveDir)
runExp(tracks, clusterTracksByKMeans_2, '_kMeans_5Feat', saveDir)
runExp(tracks, clusterTracksByFNMTF_DOCA, '_FNMTF_doca', saveDir)
runExp(tracks, clusterTracksByFNMTF_ZIP, '_FNMTF_zip', saveDir)
runExp(tracks, clusterTracksByNMF_DOCA, '_NMF_doca', saveDir)
runExp(tracks, clusterTracksByNMF_ZIP, '_NMF_zip', saveDir)
