# This script contains functions to analyze the results from the AllTracksVsCoreTracksEXP script
# Author: Alan Joshua Aneeth Jegaraj

import pandas as pd
import numpy as np
from Code.prepareData import loadFile


def getStats(vals, title='', prin=True, onlyGetTheseEvents=None, minPVSize=5):
    gtTotalTracks = []
    foundTracks = []
    averagePercentage = 0
    clustersLabelledFound = 0
    totalClusters = 0

    for eventId in vals.keys():

        if onlyGetTheseEvents is None:
            for gtId in vals[eventId]['data']['pv'].keys():
                if vals[eventId]['data']['pv'][gtId]['totalTracksInPV'] >= minPVSize:
                    gtTotalTracks.append(vals[eventId]['data']['pv'][gtId]['totalTracksInPV'])

                    foundT = vals[eventId]['data']['pv'][gtId]['found']['tracksFound']
                    if foundT > 0:
                        foundTracks.append(foundT)
                    else:
                        foundTracks.append(0)
                    averagePercentage += foundTracks[-1] / gtTotalTracks[-1]

            for clId, cl in vals[eventId]['data']['cluster'].items():
                totalClusters += 1
                if cl['found'] != -1:
                    clustersLabelledFound += 1

        elif eventId in onlyGetTheseEvents:
            for gtId in vals[eventId]['data']['pv'].keys():
                if vals[eventId]['data']['pv'][gtId]['totalTracksInPV'] >= minPVSize:
                    gtTotalTracks.append(vals[eventId]['data']['pv'][gtId]['totalTracksInPV'])

                    foundT = vals[eventId]['data']['pv'][gtId]['found']['tracksFound']
                    if foundT > 0:
                        foundTracks.append(foundT)
                    else:
                        foundTracks.append(0)
                    averagePercentage += foundTracks[-1] / gtTotalTracks[-1]

            for clId, cl in vals[eventId]['data']['cluster'].items():
                totalClusters += 1
                if cl['found'] != -1:
                    clustersLabelledFound += 1

    averagePercentage /= len(foundTracks)

    zeros = 0
    for v in foundTracks:
        if v == 0:
            zeros += 1

    reconstructionEfficiency = totalClusters / len(gtTotalTracks)
    falsePositiveRate = (totalClusters - clustersLabelledFound) / totalClusters
    if prin:
        print(f'{title}')
        print(f'Total number of PVs : {len(gtTotalTracks)}')
        print(f'PVs discovered: {len(gtTotalTracks) - zeros}')
        print(f'Percentage PVs discovered: {((len(gtTotalTracks) - zeros) * 100.0) / len(gtTotalTracks)}%')
        print(f'Average percentage tracks discovered per PV: {averagePercentage * 100.0}%')
        print(f'Total Number of clusters: {totalClusters}')
        print(f'Number of Clusters labelled as found: {clustersLabelledFound}')
        print(f'Ratio between num of clusters found and total clusters: {clustersLabelledFound/totalClusters}')
        print(f'Reconstruction efficiency: {reconstructionEfficiency}')
        print(f'False positive rate: {falsePositiveRate}')
        print('-----------------------------------')

    return {'exp': title, 'total PVs': len(gtTotalTracks), 'PVs found': len(gtTotalTracks) - zeros,
            'Percentage Pvs found %': ((len(gtTotalTracks) - zeros) * 100.0) / len(gtTotalTracks),
            'Average Percentage Tracks found per PV %': averagePercentage * 100.0,
            'Total Number of clusters': totalClusters, 'Number of Clusters labelled as found': clustersLabelledFound,
            'Ratio between num of clusters found and total clusters': clustersLabelledFound/totalClusters,
            'Reconstruction Efficiency ': reconstructionEfficiency, 'False positive rate': falsePositiveRate}


def getResultsPerEvent(data):
    table = []
    for gtId in data.keys():
        table.append([gtId, data[gtId]["count"], data[gtId]['totalTracksInPV'],
                      data[gtId]['totalTracksFound'], data[gtId]["percentageTracksFound"]])

    tableWithHeader = {'PV ID': [], 'Number_of_discovered_PVs_within_500_microns': [], 'total_tracks_in_PV': [],
                       'total_tracks_found': [], 'percentage_tracks_found %': []}

    for index, head in enumerate(tableWithHeader.keys()):
        for row in table:
            tableWithHeader[head].append(row[index])

    dat = pd.DataFrame(tableWithHeader)
    dat = dat.set_index('PV ID')
    dat = dat.sort_values('PV ID')
    return dat


def compileResultsInTable_fromDisk(files, dir, onlyGetTheseEvents=None):
    table = []
    for fileName in files:
        res = getStats(loadFile(f'{dir}{fileName}.pickle')['events'], f'{fileName}', False, onlyGetTheseEvents=onlyGetTheseEvents)
        table.append(res)

    headers = table[0].keys()
    vals = {}
    for key in headers:
        v = []
        for tab in table:
            v.append(tab[key])
        vals[key] = v

    return pd.DataFrame(vals)


# Compile different experiment results into one table
def compileResultsInTable(expResults, expNames, onlyGetTheseEvents=None):
    if len(expResults) != len(expNames):
        raise ValueError("The number of experiment names and column must be equal")
    table = []
    for val, expName in zip(expResults, expNames):
        res = getStats(val['events'], expName, False, onlyGetTheseEvents=onlyGetTheseEvents)
        table.append(res)

    headers = table[0].keys()
    vals = {}
    for key in headers:
        v = []
        for tab in table:
            v.append(tab[key])
        vals[key] = v

    return pd.DataFrame(vals)


# This function outputs a pandas dataframe that compiles results for multiple experiments based on how they perform on \
# different PV types

# keyTable = a pandas dataframe that labels all PVs in all events by their PV characteristic
def compileResultsByPVCombination(expValues, compNames, keyTable):

    if len(expValues) != len(compNames):
        raise ValueError("The number of experiment names and column must be equal")

    # all combinations
    combinations = [[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]]

    def getIdsByCombination(table, minPVSize=5):
        idsByComb = {}
        for comb in combinations:
            idPVs = table[(table['Number Of Tracks label'] == comb[0]) &
                          (table['Variance label'] == comb[1]) &
                          (table['Distance to closest PV label'] == comb[2])]

            idsByComb[f'{comb[0]}{comb[1]}{comb[2]}'] = (
                idPVs[idPVs['Number Of Tracks'] >= minPVSize]['Event:PV id']).to_numpy()
            print(comb, f': {len(idsByComb[f"{comb[0]}{comb[1]}{comb[2]}"])}')
        return idsByComb

    idsByComb = getIdsByCombination(keyTable)

    resultsByComb = {'combination': [], 'Total PVs': []}
    for col in compNames:
        resultsByComb[f'PVs found by {col}'] = []
        resultsByComb[f'Percentage PVs found by {col}'] = []

    for comb in combinations:

        combStr = f"{comb[0]}{comb[1]}{comb[2]}"
        foundCount = np.zeros(len(expValues))

        for id in idsByComb[combStr]:
            eventId = int(float(id.split(':')[0]))
            pvId = int(float(id.split(':')[1]))

            for index in range(len(expValues)):
                if expValues[index][eventId]['data']['pv'][pvId]['found']['clId'] > 0: foundCount[index] += 1

        totalPVs = len(idsByComb[f"{comb[0]}{comb[1]}{comb[2]}"])
        combExcelHeader = str(combStr).replace('0', 'L').replace('1', 'H')
        resultsByComb['combination'].append(combExcelHeader)
        resultsByComb['Total PVs'].append(totalPVs)

        for index in range(len(expValues)):
            resultsByComb[f'PVs found by {compNames[index]}'].append(foundCount[index])
            resultsByComb[f'Percentage PVs found by {compNames[index]}'].append(foundCount[index] * 100.0 / totalPVs)

    finalResults = pd.DataFrame(resultsByComb)
    return finalResults
