# This scripts plots stacked bar plots for HAC and FNMTF clustering on all-tracks and only core-tracks
# Author: Alan Joshua Aneeth Jegaraj

from PV_methods import *

saveDir = '../Exp_results/tracksTest/'  # Results save location

tracks = loadAndPrepareAllEvents('../Data/', 500, 1)  # load event (Must be only one)

# plots stacked bar plots for HAC clustering for both all-tracks and core-tracks
plotClusterStackedBarPlots(tracks, 0, clusterTracksByHAC, 'HAC')

# plots stacked bar plots for FNMTF clustering for both all-tracks and core-tracks
plotClusterStackedBarPlots(tracks, 0, clusterTracksByFNMTF_ZIP, 'FNMTF_ZIP')
