# This script plots the ZIP density plot of each PV, for both all-tracks and only-core


from PV_methods import *

eventFile = 500
dataDir = '../Data/'
numEventsToLoad = 1

print('loading data...')
tracks = loadAndPrepareAllEvents(dataDir, eventFile, numEventsToLoad)
print('finished loading')

fig = plt.figure(figsize=(15, 10), dpi=100)
ax = fig.add_subplot(111)

plotDensityPlotForPVZIPDistribution(tracks, ax)  # Density plot of all tracks
plotDensityPlotForPVZIPDistribution(tracks[tracks.track_label == -1], ax)  # Density plot of only core-tracks

plt.xlim(-50, 200)
plt.title('PV density distribution (all-tracks[-] vs core-tracks[--])', fontsize='x-large')
plt.xlabel('ZIP', fontsize='x-large')
plt.ylabel('Density', fontsize='x-large')

ax.get_legend().remove()
plt.show()
