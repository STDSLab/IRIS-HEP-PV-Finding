from PV_Experiments import *

print('loading events...')
labelled = loadAndPrepareAllEvents('../Data/', 1000)  # Loads events, and labels tracks
print('finished loading events')

dir = '../Exp_results/tracksTest/'  # directory for saving results and plots

print('started allTracks...')

# CODE FOR ALL TRACKS

# Clusters tracks and calculates the percentage of tracks found for each PV
allTracks = clusterAndCalculatePercentageTracksFound(labelled, debug=True)

saveResults(allTracks, f'{dir}allTracks.pickle')  # Saves results to file
plotPVTrackCountVsFoundTracks(allTracks, 'All tracks', f'{dir}allTracksPlot.png')  # Plots and saves it to file

print('started only Core...')


# CODE FOR ONLY CORE TRACKS

# Clusters tracks and calculates the percentage of tracks found for each PV (only core tracks are given as input)
onlyCore = clusterAndCalculatePercentageTracksFound(labelled[labelled.track_label == -1], debug=True)
saveResults(onlyCore, f'{dir}coreTracks.pickle')  # Saves results to file
plotPVTrackCountVsFoundTracks(onlyCore, 'Only core tracks', f'{dir}coreTracksPlot.png')  # Plots and saves it to file

print('finished')
