# This script contains functions to analyze the results from the AllTracksVsCoreTracksEXP script
# Author: Alan Joshua Aneeth Jegaraj

from pandas import ExcelWriter

from SGC_diffHypParamExperiments import *


def getStats(vals, title, prin=True):
    gtTotalTracks = []
    foundTracks = []
    averagePercentage = 0
    for eventId in vals.keys():
        for gtId in vals[eventId].keys():
            gtTotalTracks.append(vals[eventId][gtId]['totalTracksInPV'])
            foundTracks.append(vals[eventId][gtId]['totalTracksFound'])
            averagePercentage += foundTracks[-1] / gtTotalTracks[-1]
    averagePercentage /= len(foundTracks)

    zeros = 0
    for v in foundTracks:
        if v == 0:
            zeros += 1

    if prin:
        print(f'{title}')
        print(f'Total number of PVs : {len(gtTotalTracks)}')
        print(f'PVs discovered: {len(gtTotalTracks) - zeros}')
        print(f'Percentage PVs discovered: {((len(gtTotalTracks) - zeros) * 100.0) / len(gtTotalTracks)}%')
        print(f'Average percentage tracks discovered per PV: {averagePercentage * 100.0}%')
        print('-----------------------------------')

    return {'exp': title, 'total PVs': len(gtTotalTracks), 'PVs found': len(gtTotalTracks) - zeros,
            'Percentage Pvs found %': ((len(gtTotalTracks) - zeros) * 100.0) / len(gtTotalTracks),
            'Average Percentage Tracks found per PV %': averagePercentage * 100.0}


def getTotalTracksPerPV(fileName):
    vals = loadFile(fileName)
    gtTotalTracks = []
    for eventId in vals.keys():
        for gtId in vals[eventId].keys():
            gtTotalTracks.append(vals[eventId][gtId]['totalTracksInPV'])
    return gtTotalTracks


def outputTextAndPlot(fileName, dir, window=5, save=False):
    allTracks = loadFile(f'{dir}allTracks{fileName}.pickle')
    onlyCore = loadFile(f'{dir}coreTracks{fileName}.pickle')

    getStats(allTracks, f'all-tracks{fileName}')
    getStats(onlyCore, f'core-tracks{fileName}')

    # Changes the values so that onlyCore PVs and all-track PVs include all tracks in their count,
    # as so far only-core only keeps track of core tracks
    for eventId in onlyCore.keys():
        for gtId in onlyCore[eventId].keys():
            onlyCore[eventId][gtId]['totalTracksInPV'] = allTracks[eventId][gtId]['totalTracksInPV']

    plotGTTrackCountVsDiscoveredFraction(allTracks, f'All tracks-{fileName}',
                                         f'{dir}allTracksPlot_{fileName}.png', window=window, save=save)

    plotGTTrackCountVsDiscoveredFraction(onlyCore, f'Core tracks-{fileName}',
                                         f'{dir}coreTracksPlot_{fileName}.png', window=window, save=save)


def compileResultsInTable(files, dir):
    table = []
    for fileName in files:
        res1 = getStats(f'{dir}allTracks{fileName}.pickle', f'all-tracks{fileName}', False)
        res2 = getStats(f'{dir}coreTracks{fileName}.pickle', f'core-tracks{fileName}', False)

        table.append(res1)
        table.append(res2)

    headers = table[0].keys()
    vals = {}
    for key in headers:
        v = []
        for tab in table:
            v.append(tab[key])
        vals[key] = v

    return pd.DataFrame(vals)


dir = '../Exp_results/tracksTest/'

# window = 5
# fileName = '_HAC'
#
# allTracks = loadFile(f'{dir}allTracks{fileName}.pickle')
# onlyCore = loadFile(f'{dir}coreTracks{fileName}.pickle')
# values = []
#
# for eventId in allTracks.keys():
#     allT = getResultsPerEvent(allTracks[eventId])
#     coreT = getResultsPerEvent(onlyCore[eventId])
#
#     diff = sum(coreT['Number_of_discovered_PVs_within_500_microns'] != 0) - \
#            sum(allT['Number_of_discovered_PVs_within_500_microns'] != 0)
#     values.append(diff)
#
# values = np.array(values)
#
# print(f'{fileName} average difference: {sum(values)}')
# print(f'{fileName} Number of events where PVs are being found less: {len(values[values<0])}')
# print(f'{fileName} Number of events where PVs are being found more: {len(values[values>0])}')
# print(f'{fileName} Number of events where PVs found did not change: {len(values[values==0])}')
#
# x,y = concatValuesIntoBins(list(range(len(values))), values, window)
# plotBarChart(x,y,f'Change in number of PVs discovered{fileName}','Event','Change', width=window, color='royalblue')
#
# fileName = '_FNMTF_ZIP'
#
# allTracks = loadFile(f'{dir}allTracks{fileName}.pickle')
# onlyCore = loadFile(f'{dir}coreTracks{fileName}.pickle')
# values = []
#
# for eventId in allTracks.keys():
#     allT = getResultsPerEvent(allTracks[eventId])
#     coreT = getResultsPerEvent(onlyCore[eventId])
#
#     diff = sum(coreT['Number_of_discovered_PVs_within_500_microns'] != 0) - \
#            sum(allT['Number_of_discovered_PVs_within_500_microns'] != 0)
#     values.append(diff)
#
# values = np.array(values)
# print(f'{fileName} average difference: {sum(values)}')
# print(f'{fileName} Number of events where PVs are being found less: {len(values[values<0])}')
# print(f'{fileName} Number of events where PVs are being found more: {len(values[values>0])}')
# print(f'{fileName} Number of events where PVs found did not change: {len(values[values==0])}')
#
# x,y = concatValuesIntoBins(list(range(len(values))), values, window)
# plotBarChart(x,y,f'Change in number of PVs discovered{fileName}','Event','Change', width=window, color='royalblue')


# Plot histogram of PV track count
# trackCounts = getTotalTracksPerPV(f'{dir}allTracks_HAC.pickle')
# plt.figure()
# plt.hist(trackCounts,bins=100)
# plt.title('PV track count histogram')
# plt.xlabel('track count')
# plt.ylabel('Number of PVs')
# plt.show()

outputTextAndPlot('_HAC', dir, save=True)
outputTextAndPlot('_HAC_5Feat', dir, save=True)
outputTextAndPlot('_kMeans', dir, save=True)
outputTextAndPlot('_kMeans_5Feat', dir, save=True)
outputTextAndPlot('_NMF_zip', dir, save=True)
outputTextAndPlot('_NMF_doca', dir, save=True)
outputTextAndPlot('_FNMTF_zip', dir, save=True)
outputTextAndPlot('_FNMTF_doca', dir, save=True)

# files = ['_HAC', '_HAC_5Feat', '_kMeans', '_kMeans_5Feat', '_FNMTF_zip', '_FNMTF_doca', '_NMF_zip', '_NMF_doca']
# res = compileResultsInTable(files, dir)
# # saveResults(res,f'{dir}TABLE.pickle')
#
# plotBarChart(res['exp'], res['PVs found'], title='PVs found', yLabel=f'PVs found (out of {res["total PVs"][0]})',
#              labelRotation='vertical', color=genColorList(len(res['exp']) / 2, elemRepeatCount=2), save=True,
#              fileName=f'{dir}PVsFound.png')
