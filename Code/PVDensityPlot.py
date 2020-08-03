# This script plots the ZIP density plot of each PV, for both all-tracks and only-core
# Author: Alan Joshua Aneeth Jegaraj


from PV_methods import *

eventFile = 500
dataDir = '../Data/'
numEventsToLoad = 6
eventInd = 5
saveFile = f'../Exp_results/tracksTest/PVDensityPlotAllVsCore_Event_{eventInd}.png'

print('loading data...')
tracks = loadAndPrepareAllEvents(dataDir, eventFile, numEventsToLoad)
print('finished loading')

tracks_event = tracks[tracks.eventId == eventInd]

fig = plt.figure(figsize=(15, 10), dpi=100)
ax = fig.add_subplot(111)

# Density plot of all tracks
labels1 = plotDensityPlotForPVZIPDistribution(tracks_event, ax, style='-', labelPrefix='all-tracks')
# Density plot of only core-tracks
labels2 = plotDensityPlotForPVZIPDistribution(tracks_event[tracks_event.track_label == -1], ax, style='--', labelPrefix='core'
                                                                                                            '-tracks')

plt.xlim(-50, 400)
plt.title(f'PV density distribution Event:{eventInd}', fontsize='x-large')
plt.xlabel('ZIP', fontsize='x-large')
plt.ylabel('Density', fontsize='x-large')

legend = []
legend.extend(labels1)
legend.extend(labels2)

ax.legend(legend)
plt.savefig(saveFile, dpi=300)
plt.show()
