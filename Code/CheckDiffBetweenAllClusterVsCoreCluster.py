from PV_Experiments import *

print('loading events...')
labelled = loadAndPrepareAllEvents('../Data/', 200, 1)
print('finished loading events')

gtPvs = len(labelled['gt'].unique())
clusteredTracks_HAC = clusterTracksByHAC(labelled, gtPvs)

found_centroids, gt_centroids = calculateCentroidForFoundAndGTClusters(clusteredTracks_HAC)
res = calcPercentageTracksFound(found_centroids, gt_centroids)

print('---------------------')
print('All tracks:')
printClusterResults(res)
print('---------------------')
print('---------------------')

clusteredTracks_HAC_core = clusterTracksByHAC(labelled[labelled['track_label'] == -1], gtPvs)

found_centroids_core, gt_centroids_core = calculateCentroidForFoundAndGTClusters(clusteredTracks_HAC_core)
res_core = calcPercentageTracksFound(found_centroids_core, gt_centroids_core)

print('Only core tracks:')
printClusterResults(res_core)
