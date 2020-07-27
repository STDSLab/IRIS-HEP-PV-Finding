import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import uproot
from scipy import stats
import sklearn as sk
from sklearn import cluster as cl
from scipy.sparse import csr_matrix
import random
import copy

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

from spektral.datasets import citation
from spektral.layers import GraphConv
from spektral.utils.convolution import localpooling_filter

from scipy.sparse import csr_matrix
from keras import backend as KBack
import itertools

from sklearn import cluster as cl
from sklearn import neighbors


def z2coord(cPos, cTraj, z, z0):
    return cPos + cTraj * (z - z0)


def z2xy(xyPos, xyTraj, z, z0):
    return (z2coord(xyPos[0], xyTraj[0], z, z0), z2coord(xyPos[1], xyTraj[1], z, z0))


def coord2spherical(x, y, z):
    '''
    Compute spherical coordinates (physics)

    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    z : float
        z coordinate.

    Returns
    -------
    rho : float
        Radius of sphere.
    phi : float
        Azimuthal angle.
    theta : float
        Polar angle.

    '''
    return np.sqrt(x ** 2 + y ** 2 + z ** 2), np.arctan2(y, x), np.arctan2(np.sqrt(x ** 2 + y ** 2), z)


def coord2polar(x, y):
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)


# Based on cosTheta in https://root.cern.ch/doc/master/TVector3_8h_source.html#l00371
def cosTheta(mag, z):
    if mag == 0:
        return 1.0
    return z / mag


# Based on eta in https://root.cern.ch/doc/master/TVector3_8h_source.html#l00400
#  which uses PseudoRapidity https://root.cern.ch/doc/master/TVector3_8cxx_source.html#l00320
def eta(mag, z):
    ct = cosTheta(mag, z)
    if ct ** 2 < 1:
        return -0.5 * np.log((1 - ct) / (1 + ct))

    if z == 0:
        return 0
    if z > 0:
        return 10e10
    return -10e10


# %% particle reconstructability
# based on ntrkInAcc in fcn.h https://gitlab.cern.ch/LHCb-Reco-Dev/pv-finder/-/blob/master/ana/fcn.h
def isReconstructable(pNmHits, ppx, ppy, ppz, pz, pvz):
    # Determines if particle is reconstructable
    # if hits < 3
    # if difference between particle z and pv z > 0.001
    # if particle eta < 2 or > 5
    # if particle rho < 3

    if pNmHits < 3:
        return False
    if np.abs(pz - pvz) > 0.001:
        return False

    mag = coord2spherical(ppx, ppy, ppz)[0]
    etaV = eta(mag, ppz)
    if etaV < 2 or etaV > 5:
        return False
    if mag < 3:
        return False

    return True


def computeZ_ip(z1, z2, r1, r2):
    s = (r2 - r1) / (z2 - z1)
    z0 = r1 - s * z1
    return -(z0 / s)


# points are dictionaries {'x':#, 'y':#, 'z':#}
def coordCompZ_ip(p1, p2):
    r1, phi1 = coord2polar(p1['x'], p1['y'])
    r2, phi2 = coord2polar(p2['x'], p2['y'])

    return computeZ_ip(p1['z'], p2['z'], r1, r2)


class importer:
    def __init__(self, ns, dp: Path):
        self._nmSims = ns
        self._dataPth = dp

    # import data where nm = file name and objNm = tag
    def _impData(self, nm, objNm):
        # change the below line if your file name is different than the standard [nm]_#pvs.root naming scheme
        dPth = self._dataPth.joinpath(nm + '_' + str(self._nmSims) + 'pvs.root')
        rootObj = uproot.open(dPth)
        return rootObj[objNm]

    def extractData(self, iSim):
        dataObj = self._impData('pv', 'data')
        tksDataObj = self._impData('trks', 'trks')
        nmSims = dataObj.lazyarrays().size

        dataVals = dataObj.lazyarrays()[iSim]
        tksDataVals = tksDataObj.lazyarrays()[iSim]

        sim, tk = {}, {}
        if iSim > nmSims or iSim < 0:
            print('Out of bounds')
            return sim, tk

        sim['hits'] = {'x': dataVals['hit_x'], 'y': dataVals['hit_y'], 'z': dataVals['hit_z'],
                       'prt': dataVals['hit_prt']}

        sim['pvs'] = {'x': dataVals['pvr_x'],
                      'y': dataVals['pvr_y'],
                      'z': dataVals['pvr_z']}

        sim['svs'] = {'x': dataVals['svr_x'],
                      'y': dataVals['svr_y'],
                      'z': dataVals['svr_z'],
                      'pv': dataVals['svr_pvr']}

        sim['prts'] = {'x': dataVals['prt_x'],
                       'y': dataVals['prt_y'],
                       'z': dataVals['prt_z'],
                       'px': dataVals['prt_px'],
                       'py': dataVals['prt_py'],
                       'pz': dataVals['prt_pz'],
                       'e': dataVals['prt_e'],
                       'id': dataVals['prt_pid'],
                       'hit': dataVals['prt_hits'],
                       'pv': dataVals['prt_pvr']}

        # compute reconstructability for each particle
        prtRec = []
        nmPrt = sim['prts']['x'].size
        for iPrt in range(0, nmPrt):
            nmHits = np.where(sim['hits']['prt'] == iPrt)[0].size
            prtRec.append(
                isReconstructable(
                    nmHits,
                    sim['prts']['px'][iPrt],
                    sim['prts']['py'][iPrt],
                    sim['prts']['pz'][iPrt],
                    sim['prts']['z'][iPrt],
                    sim['pvs']['z'][int(sim['prts']['pv'][iPrt])]))

        #             print(sim['prts']['z'][iPrt])
        #             print(sim['pvs']['z'][int(sim['prts']['pv'][iPrt])])
        #             print(prtRec[-1])
        #             print(nmHits)
        #             print('-----')

        sim['prts']['recAbility'] = np.array(prtRec)

        tk['pos'] = {'x': tksDataVals['recon_x'],
                     'y': tksDataVals['recon_y'],
                     'z': tksDataVals['recon_z']}

        tk['hits'] = {'x1': tksDataVals['hit1_x'],
                      'x2': tksDataVals['hit2_x'],
                      'x3': tksDataVals['hit3_x'],
                      'y1': tksDataVals['hit1_y'],
                      'y2': tksDataVals['hit2_y'],
                      'y3': tksDataVals['hit3_y'],
                      'z1': tksDataVals['hit1_z'],
                      'z2': tksDataVals['hit2_z'],
                      'z3': tksDataVals['hit3_z']}

        tk['traj'] = {'x': tksDataVals['recon_tx'],
                      'y': tksDataVals['recon_ty'],
                      'chi2': tksDataVals['recon_chi2']}

        return sim, tk


# %% gen tracks code
# generate tracks using ground truth particles
# eventIdx is used to track which event a track belongs to.
# -1 is the default eventId, meaning it doesn't belong anywhere
def genGTTracks(sim, eventIdx=-1):
    prtTracks = {}
    nmPrts = sim['prts']['x'].size
    for prt in range(0, nmPrts):
        prtTk = {}
        # assignment of hits to tracks
        # always returns the hits in order of closest to beam line
        hitIdx = np.where(sim['hits']['prt'] == prt)[0]
        for key in ['x', 'y', 'z']:
            prtTk[key] = sim['hits'][key][hitIdx]

        prtTk['gt'] = sim['prts']['pv'][prt]
        prtTk['eventId'] = eventIdx

        pv = {'x': sim['pvs']['x'][int(prtTk['gt'])],
              'y': sim['pvs']['y'][int(prtTk['gt'])], 'z': sim['pvs']['z'][int(prtTk['gt'])]}
        prtTk['pv'] = pv

        # error computation
        # assume first hit is 100% accurate, can draw a track between the pv and first hit
        # error is then computed as the average hit distance away from this line
        pvIdx = int(sim['prts']['pv'][prt])
        x0 = sim['pvs']['x'][pvIdx]
        y0 = sim['pvs']['y'][pvIdx]
        z0 = sim['pvs']['z'][pvIdx]

        zDiff = prtTk['z'][0] - z0
        px = (prtTk['x'][0] - x0) / zDiff
        py = (prtTk['y'][0] - y0) / zDiff

        prtTk['trajX'] = px
        prtTk['trajY'] = py

        # Angle in RZ calculation
        if len(prtTk['x']) > 1 and len(prtTk['y']) > 1 and len(prtTk['z']) > 1:
            r1, phi1 = coord2polar(prtTk['x'][0], prtTk['y'][0])
            z1 = prtTk['z'][0]
            r2, phi2 = coord2polar(prtTk['x'][1], prtTk['y'][1])
            z2 = prtTk['z'][1]

            delZ = z2 - z1
            delX = r2 - r1
            angle = np.arctan(delX / delZ)
            prtTk['angle'] = angle
        else:
            prtTk['angle'] = np.pi / 2

        # if there is only one hit, FOR NOW ignore
        nmHits = prtTk['x'].size
        if nmHits < 2:
            # prtTk['error'] = 0
            continue
        else:
            # TODO change error computation to magnitude
            cmbErr = 0
            for hit in range(1, nmHits):
                (lx, ly) = z2xy([x0, y0], [px, py], prtTk['z'][hit], z0)
                cmbErr += np.sqrt((prtTk['x'][hit] - lx) ** 2 + (prtTk['y'][hit] - ly) ** 2)
            prtTk['error'] = cmbErr / (nmHits - 1)

        prtTracks[prt] = prtTk
    return prtTracks


# generate tracks using reconstructed tracks
# eventIdx is used to track which event a track belongs to.
# -1 is the default eventId, meaning it doesn't belong anywhere
def genRecTracks(tk, eventIdx=-1):
    tkTracks = {}
    nmTks = tk['pos']['x'].size
    for prt in range(0, nmTks):
        trkTk = {}

        # assignment of hits to tracks
        nmHits = 2
        for key in ['x', 'y', 'z']:
            trkTk[key] = np.array([tk['hits'][key + '1'][prt], tk['hits'][key + '2'][prt]])

        # TODO associate rec track with gt PV

        # error computation
        # as the track reconstruction module returns px and py we can draw a line based on the predicted track
        # compute error as the average hit distance away from this line
        x0 = tk['pos']['x'][prt]
        y0 = tk['pos']['x'][prt]
        z0 = tk['pos']['x'][prt]

        px = tk['traj']['x'][prt]
        py = tk['traj']['y'][prt]

        trkTk['trajX'] = px
        trkTk['trajY'] = py
        trkTk['eventId'] = eventIdx

        # Angle in RZ calculation
        if len(trkTk['x']) > 1 and len(trkTk['y']) > 1 and len(trkTk['z']) > 1:
            r1, phi1 = coord2polar(trkTk['x'][0], trkTk['y'][0])
            z1 = trkTk['z'][0]
            r2, phi2 = coord2polar(trkTk['x'][1], trkTk['y'][1])
            z2 = trkTk['z'][1]

            delZ = z2 - z1
            delX = r2 - r1
            angle = np.arctan(delX / delZ)
            trkTk['angle'] = angle
        else:
            trkTk['angle'] = np.pi / 2

        cmbErr = 0
        for hit in range(0, nmHits):
            (lx, ly) = z2xy([x0, y0], [px, py], trkTk['z'][hit], z0)
            cmbErr += np.sqrt((trkTk['x'][hit] - lx) ** 2 + (trkTk['y'][hit] - ly) ** 2)
        trkTk['error'] = cmbErr / nmHits

        # TODO set key equal to most similar gt track
        # currently uses reconstructed track ID
        tkTracks[prt] = trkTk
    return tkTracks


# computes Z_ip for each track and returns as a dictionary where the Z_ip value is assigned to each track ID
def genZip(tkData):
    tkZip = {}
    # generate track Z_ip by looping through all the tracks
    for tkKey in tkData.keys():
        # print(tkKey)
        tkZip[tkKey] = coordCompZ_ip(
            {'x': tkData[tkKey]['x'][0],
             'y': tkData[tkKey]['y'][0],
             'z': tkData[tkKey]['z'][0]},
            {'x': tkData[tkKey]['x'][1],
             'y': tkData[tkKey]['y'][1],
             'z': tkData[tkKey]['z'][1]})

    return tkZip


def findPoint(x, p, v):
    return p + x * v


# assumes two lines P and Q each defined by two points 0 and 1
def findDOCA(P0, P1, Q0, Q1):
    u = P1 - P0
    v = Q1 - Q0
    w0 = P0 - Q0

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    pVal = a * c - b ** 2

    if pVal > 0.00000001:
        x_p = (b * e - c * d) / pVal
        x_q = (a * e - b * d) / pVal
    else:
        # is parallel or almost parallel
        x_p = 0
        if b > c:
            x_q = d / b
        else:
            x_q = e / c

    PC = findPoint(x_p, P0, u)
    QC = findPoint(x_q, Q0, v)

    return PC, QC, np.linalg.norm(PC - QC)


# TODO optimize
# computes the DOCA matrix for all pairs of tracks
def genPOCADist(tkData):
    tkKeys = list(tkData)
    nmTks = len(tkKeys)
    # if DOCA is greater than 1 mm away from the center beam line (x or y outside of the -1, 1 range),
    #  set to some high number. Disable by setting to np.inf
    maxDist = 1.0

    distMat = np.zeros((nmTks, nmTks))
    for iTk, tkKey in enumerate(tkKeys):
        # compute DOCA
        for iTk2, tkKey2 in enumerate(tkKeys[iTk:]):
            if tkKey != tkKey2:
                tk = tkData[tkKey]
                tk2 = tkData[tkKey2]
                pointSet = \
                    np.array([[tk['x'][0], tk['y'][0], tk['z'][0]], \
                              [tk['x'][1], tk['y'][1], tk['z'][1]], \
                              [tk2['x'][0], tk2['y'][0], tk2['z'][0]], \
                              [tk2['x'][1], tk2['y'][1], tk2['z'][1]]])

                P1, P2, dist = \
                    findDOCA(pointSet[0, :], pointSet[1, :], \
                             pointSet[2, :], pointSet[3, :])

                P1OutFlag = np.logical_or(P1 > maxDist, P1 < -maxDist)
                P2OutFlag = np.logical_or(P2 > maxDist, P2 < -maxDist)

                if np.count_nonzero(P1OutFlag[0:2]) + \
                        np.count_nonzero(P2OutFlag[0:2]) > 0:
                    dist = dist + 1000

                distMat[iTk, iTk2 + iTk] = dist

    distMat = distMat + np.transpose(distMat)

    return distMat


def getPOCABetween2Tracks(tkData, key1, key2):
    maxDist = 1.0
    tk = tkData[key1]
    tk2 = tkData[key2]
    pointSet = \
        np.array([[tk['x'][0], tk['y'][0], tk['z'][0]], \
                  [tk['x'][1], tk['y'][1], tk['z'][1]], \
                  [tk2['x'][0], tk2['y'][0], tk2['z'][0]], \
                  [tk2['x'][1], tk2['y'][1], tk2['z'][1]]])

    P1, P2, dist = \
        findDOCA(pointSet[0, :], pointSet[1, :], \
                 pointSet[2, :], pointSet[3, :])
    return P1, P2


def addZipToTracks(tracks, ZIP):
    newTk = tracks
    for key in ZIP.keys():
        newTk[key]['zip'] = ZIP[key]
    return newTk


def dictArrayToArray(dictArray, dtype=float):
    return np.fromiter(
        dictArray.values(), dtype, count=len(dictArray))


def removeGlobalOutliersByIQR(tracks, IQRThresh=1.5):
    finalTracks = pd.DataFrame()
    uniqueEventIds = tracks['eventId'].unique()

    for eventId in uniqueEventIds:
        currEventTracks = tracks[tracks['eventId'] == eventId]
        currEventTracks = currEventTracks.transpose()

        Q1 = currEventTracks.loc['zip', :].quantile(0.25)
        Q3 = currEventTracks.loc['zip', :].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((currEventTracks.loc['zip'] < (Q1 - IQRThresh * IQR)) | (
                currEventTracks.loc['zip'] > (Q3 + IQRThresh * IQR)))
        finalTracks = finalTracks.append((currEventTracks.loc[:, mask]).transpose())

    return finalTracks


def removeGlobalOutliersByZScore(tracks, ZThresh=1.5):
    finalTracks = pd.DataFrame()
    uniqueEventIds = tracks['eventId'].unique()

    for eventId in uniqueEventIds:
        currEventTracks = (tracks[tracks['eventId'] == eventId]).transpose()
        trackKeys = list(currEventTracks.columns.values)
        zScores = np.abs(stats.zscore([val for val in currEventTracks.loc['zip']]))
        zScoresDict = {}
        for index, val in enumerate(zScores):
            zScoresDict[trackKeys[index]] = val

        currEventTracks = currEventTracks.append(pd.Series(data=zScoresDict, name='zScore'), ignore_index=False)
        mask = (currEventTracks.loc['zScore'] < ZThresh)
        finalTracks = finalTracks.append(((currEventTracks.loc[:, mask]).drop(['zScore'])).transpose())

    return finalTracks


# List of tuples, where the first element of each tuple contains the cluster ID and the second element contains the track
# Helper function used by clustering methods
def addClusterKeyToTracks(pairs):
    tracks = {}
    for pair in pairs:
        tracks[int(pair[1].name)] = pair[1].append(pd.Series([pair[0]], name='ClusterID'), ignore_index=False)
    return pd.DataFrame(tracks).rename(index={0: 'Cluster_id'})


def clusterTracksByKMeans(tracks, numClusters):
    tracks = tracks.transpose()
    vals = [[v] for v in tracks.loc['zip']]
    centroids, y_km, _ = cl.k_means(vals, init='k-means++', n_clusters=numClusters)
    pairs = [(clID, tracks[trackInd]) for clID, trackInd in zip(y_km, tracks)]
    sortedList = sorted(pairs, key=lambda x: x[0])
    return addClusterKeyToTracks(sortedList).transpose()


def clusterTracksByHAC(tracks, numClusters):
    tracks = tracks.transpose()
    vals = [[v] for v in tracks.loc['zip']]
    HAC = sk.cluster.AgglomerativeClustering(n_clusters=numClusters).fit(vals)
    pairsHAC = [(clID, tracks[trackInd]) for clID, trackInd in zip(HAC.labels_, tracks)]
    sortedList = sorted(pairsHAC, key=lambda x: x[0])
    return addClusterKeyToTracks(sortedList).transpose()


def calculateCentroidForFoundAndGTClusters(tracks):
    clusterIDs = tracks['Cluster_id'].unique()
    gtIds = tracks['gt']
    numOfFoundClusters = len(clusterIDs)

    beamLine = {'x': [0, 0], 'y': [0, 0], 'z': [-10000, 10000]}

    found_centroids = {}
    for clID in clusterIDs:
        clTracks = tracks[tracks['Cluster_id'] == clID]
        pocas = {}
        for index, row in clTracks.iterrows():
            tempTracks = {}
            tempTracks[0] = row
            tempTracks[1] = beamLine
            pocas[index] = [getPOCABetween2Tracks(tempTracks, 0, 1)]
            # print(pocas[index])

        centroid = np.array([float(0), float(0), float(0)])
        for key in pocas.keys():
            # print('-----------------')
            # print(pocas[key][0][1])
            # print('-----------------')
            centroid += np.array(pocas[key][0][1])
        centroid = centroid / len(pocas.keys())
        found_centroids[clID] = centroid

    gt_centroids = {}
    for gtID in gtIds:
        gtTracks = tracks[tracks['gt'] == gtID]

        gt_centroid = np.array([float(0), float(0), float(0)])
        for index, row in gtTracks.iterrows():
            pv = [row['pv']['x'], row['pv']['y'], row['pv']['z']]
            gt_centroid += np.array(pv)
        gt_centroid = gt_centroid / len(gtTracks.index.values)
        gt_centroids[gtID] = gt_centroid

    return found_centroids, gt_centroids


def calcPercentageTracksFound(found_clusters, gt_clusters):
    results = {}
    for gtId in gt_clusters.keys():
        count = 0
        for clID in found_clusters.keys():
            dist = np.linalg.norm(gt_clusters[gtId] - found_clusters[clID])
            if dist < 500e-3:
                count += 1
        results[gtId] = count
    return results


def printClusterResults(clusters):
    print('GT Cluster ID     ||   Number of clusters within 500 microns')
    for key in clusters.keys():
        print(f'{key}               ||   {clusters[key]} ')


def plotZIPHistogramByClusters(tracks, bins=100, title="", xLabel="", yLabel="", colorMap='gist_rainbow'):
    plt.figure()
    N, bins, patches = plt.hist(tracks['zip'], bins=bins, edgecolor='k')
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    y_km_sorted = list(tracks['Cluster_id'])

    # Unique colors code taken from https://stackoverflow.com/questions/8389636/creating-over-20
    # -unique-legend-colors-using-matplotlib
    NUM_COLORS = len(np.unique(y_km_sorted))
    cm = plt.get_cmap(colorMap)
    colorsList = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]

    for i in range(len(patches)):
        index = int((i / len(N)) * len(y_km_sorted))
        patches[i].set_facecolor(colorsList[y_km_sorted[index]])
    plt.show()


# Made by Alan
def plotStackedBarPlot(dataMatrix, labels, width=1, title='', legendLabels=[], xLabel="", yLabel="", legendOffset=0,
                       binEdges=None, align='edge'):
    fig = plt.figure()
    height = np.zeros(len(dataMatrix[0]))
    ax = fig.add_subplot(111)
    #     binEdges = np.arange(len(dataMatrix[0]))
    if (binEdges is None):
        binEdges = np.linspace(0, len(dataMatrix[0]), len(dataMatrix[0]) + 1)

    NUM_COLORS = len(dataMatrix)
    cm = plt.get_cmap(
        'gist_rainbow')  # Unique colors code taken from https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
    colorsList = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
    ax.set_prop_cycle(color=colorsList)

    for index, row in enumerate(dataMatrix):
        ax.bar(x=binEdges[:-1], height=row, bottom=height, tick_label=labels,
               width=binEdges[index + 1] - binEdges[index], edgecolor='k', align=align)
        height += np.array(row)

    colors = {}
    for index, lab in enumerate(legendLabels):
        colors[lab] = colorsList[index]

    handles = [plt.Rectangle((0, 10), 1, 1, color=colors[label], edgecolor='k') for label in legendLabels]
    ax.legend(handles, legendLabels, bbox_to_anchor=(1 + legendOffset, 0.5), loc="center left", borderaxespad=0)

    ax.set_ylim(0, list(reversed(sorted(height)))[0] + 2)  # To add extra space on top of the bars
    #     ax.set_xlim(0.5,binEdges[-1]+1)  # To add extra space on top of the bars
    ax.set_title(title, pad=20)
    ax.set_ylabel(yLabel)
    ax.set_xlabel(xLabel)
    plt.tight_layout()
    plt.show()


# Made by Kendrick
def plotStackedBar_Kendrick(multiData, binEdges, xyLbls, dataLbls,
                            redTicks=2, title="", align='edge', rotation=None,
                            barLbls=None, closeFlag=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    base = np.zeros(len(multiData[0]))
    for data, lbl in zip(multiData, dataLbls):
        ax.bar(binEdges[:-1], data, align=align, width=binEdges[1] - binEdges[0], edgecolor='k', label=lbl,
               bottom=base.tolist())

        base += np.array(data)

    pltTicks = binEdges.copy()[:-1]
    if barLbls is None:
        for i in range(0, redTicks):
            # pltTicks = redNParray(pltTicks, 2)
            pltTicks = pltTicks
        ax.set_xticks(pltTicks)
    else:
        ax.set_xticks(pltTicks)
        ax.set_xticklabels(barLbls)

    if rotation is not None:
        plt.xticks(rotation=rotation)

    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left", borderaxespad=0)
    #     plt.legend()
    ax.set_xlabel(xyLbls[0])
    ax.set_ylabel(xyLbls[1])
    ax.set_title(title)
    ax.set_ylim(0, list(reversed(sorted(base)))[0] + 2)  # To add extra space on top of the bars
    plt.tight_layout()
    plt.show()


# Helper method used in an experiment to concatenate results
# This assumes averageOverCount is a factor of the total number of elements in the incoming data

def appendAllToOneDict(history, averageOverCount=1):
    res = {}
    keys = history[0].history.keys()
    for key in keys:
        res[key] = []

    for hist in history:
        for key in hist.history.keys():
            res[key].extend(hist.history[key])

    for key in keys:
        temp = []
        offset = 0
        counter = 0
        sum = 0
        for val in res[key]:
            sum += val
            counter += 1
            if (counter == averageOverCount):
                sum /= averageOverCount
                temp.append(sum)
                sum = 0
                counter = 0
        res[key] = temp

    return res


def transposeTrackDataDictionary(matrix):
    initKeys = list(matrix.keys())
    c = len(initKeys)
    r = len(matrix[initKeys[0]])
    resMatrix = np.zeros((r, c))
    for index, key in enumerate(initKeys):
        resMatrix[:, index] = matrix[key]
    return resMatrix


# Used to generate a data structure specifically used to generate a stacked bar plot
def generateStackedBarPlotMatrixFromTracks(tracks):
    tracks = tracks.sort_values(by='gt', axis=1)  # arranges tracks in ascending order of GT cluster ID
    uniqueGTIds = np.unique(tracks.loc['gt'])
    uniqueClusterIds = np.unique(tracks.loc['Cluster_id'])
    gtTotalPVs = len(uniqueGTIds)
    totalClustersFound = len(uniqueClusterIds)

    data = {id: np.zeros(totalClustersFound) for id in uniqueGTIds}
    for trkName in tracks.columns.values:
        track = tracks[trkName]
        gtId = int(track.loc['gt'])
        clId = int(track.loc['Cluster_id'])
        data[gtId][clId] += 1
    return data


# This function is used to generate the appropriate data required for plotted a stacked bar plot from the clusters

# @Params
# plotType:
#    found-base = generates the data, including labels, to plot the found-base plot
#    gt-base = generates the data, including labels, to plot the gt-base plot
#    Note: anything other than 'found-base' would generate data for the gt-base

def genDataToPlotStackedBarPlot(tracks, plotType):
    tracks = tracks.transpose()
    data = generateStackedBarPlotMatrixFromTracks(tracks)

    if plotType == 'found-base':
        labels = np.unique(tracks.loc['Cluster_id'])
        legendLabels = list(data.keys())
        matrix = []
        for key in legendLabels:
            matrix.append(data[key])
        for index, lab in enumerate(legendLabels):
            legendLabels[index] = 'pv' + str(int(lab))
    else:
        matrix = transposeTrackDataDictionary(data)
        legendLabels = np.unique(tracks.loc['Cluster_id'])
        labels = [str(int(val)) for val in
                  list(data.keys())]  # Convert pv ID from float to int, just so that it looks better

        for index, lab in enumerate(legendLabels):
            legendLabels[index] = 'cl' + str(int(lab))

    return matrix, labels, legendLabels


# Labels all tracks as either core(-1)/non-core(0)/outlier(1) using ground truth zip distribution
def labelTracks(tracks):
    newTracks = pd.DataFrame()
    uniqueEventsIDs = tracks['eventId'].unique()

    #   Loop through every unique event
    for eventId in uniqueEventsIDs:

        currEventTrack = tracks[tracks.eventId == eventId]  # Gets current event data
        uniqueGTIds = currEventTrack['gt'].unique()  # Gets PV ids in the current event

        # Loop through tracks for each PV in current event
        for gtId in uniqueGTIds:

            # Gets tracks only associated with the current PV
            singlePVTracks = currEventTrack[currEventTrack['gt'] == gtId]
            # Gets the zScores for the zip values of the tracks in the current PV
            zScores = stats.zscore([val for val in singlePVTracks['zip']])

            # Labels each track based on the criteria:
            #   |z| <= 0.5 : core (-1)
            #   0.5 < |z| <= 1.5 : non-core (0)
            #   |z| > 1.5 : outlier (-1)

            labels = []
            for index in range(len(zScores)):
                #             print(index)
                curZScore = zScores[index]
                if -0.5 <= curZScore <= 0.5:
                    labels.append(-1)
                elif - 1.5 <= curZScore < -0.5 or 0.5 < curZScore <= 1.5:
                    labels.append(0)
                else:
                    labels.append(1)
            singlePVTracks['track_label'] = labels
            newTracks = newTracks.append(singlePVTracks)
    return newTracks


# Helper function to return select columns as python lists
# Could probably be commented out as this is not being used in the code
def convertDataFrameToFeatureMatrix(data, featuresList=['trajX', 'trajY', 'angle', 'error', 'zip']):
    return data[featuresList].values.tolist()


# This is a wrapper for "importer.extractData()" and is used to load data from multiple events at the same time

# @Params
# eventFile = The 'name' of the event file being loaded.
#             Due to the way the importer.extractData() is structured, the tracks and pvs file must both be available
#             in the specified directory.
#             It expects the particle file and tracks file to be formatted as "pv_{num}pvs.root" and
#             "trks_{num}pvs.root" respectively.
#             Therefore, the argument for eventFile would just be {num}
#             For example, when loading the pv_100pvs.root and trks_100pvs.root files,
#             the argument for eventFile would be '100'
#
# dataDir = location of the directory where the event files are located

# totalEventsToPool = Optional parameter to choose how many events to load from the specified events file
#                     By default it loads all events, but if you want to load only the first 10 events from the
#                     100 events file, the value for this parameter could be set to '10'

def loadAllEvents(eventFile, dataDir, totalEventsToPool=None):
    if totalEventsToPool is None:
        totalEventsToPool = eventFile
    allSims = []
    allTKRecs = []
    for eventID in range(totalEventsToPool):
        sim, tkRec = importer(eventFile, dataDir).extractData(eventID)
        allSims.append(sim)
        allTKRecs.append(tkRec)
    return allSims, allTKRecs


# Gets a list of simulations (each simulation holds data about one event), generates tracks from them and
# combines them into one dictionary

def poolAllGTEventTracks(sims):
    tracks = {}
    lastTrackID = 0
    for eventInd, sim in enumerate(sims):
        currTr = genGTTracks(sim, eventInd)
        for key, ind in zip(currTr.keys(), range(len(currTr))):
            tracks[ind + lastTrackID] = currTr[key]
        lastTrackID += len(currTr)
    return tracks


# Function used to generate a test graph for testing a graph generated using the graphs_neuralnet package.
# THIS IS NOT BEING USED ANYMORE, AS WE HAVE MOVED TO USING THE SPEKTRAL PACKAGE

# @ Params
# numNodes =  is a list of ints, where each int signifies the number of nodes in that class. Size of list is the
#             number of unique classes, and summation of all elements is the total number of nodes
# edgeProbabilityMatrix = is matrix which contains the probability with which nodes of different

def generateTestGraphData(numNodes, edgeProbabilityMatrix, randSeed=None, shouldAllowSelfEdge=False, nodeFeatureCount=3,
                          edgeFeatureCount=3, globalFeatureCount=1):
    nodeIndices = []
    for index, val in enumerate(numNodes):
        if index == 0:
            nodeIndices.append([num for num in range(val)])
        else:
            nodeIndices.append([num + numNodes[index - 1] for num in range(val)])

    if randSeed is not None:
        random.seed(randSeed)

    nodeFeatures = []
    for ind in range(sum(numNodes)):
        nodeFeatures.append(np.ones(nodeFeatureCount))

    senders = []
    receivers = []
    alreadyEdges = []
    for class1Ind, cl1NodesList in enumerate(nodeIndices):
        for node1 in cl1NodesList:
            for class2Ind, cl2NodesList in enumerate(nodeIndices):
                for node2 in cl2NodesList:
                    if [(node1, node2), (node2, node1)] not in alreadyEdges and (shouldAllowSelfEdge or node1 != node2):
                        prob = edgeProbabilityMatrix[class1Ind][class2Ind]
                        randNum = random.uniform(0, 1)
                        if randNum <= prob:
                            senders.append(node1)
                            receivers.append(node2)
                            receivers.append(node1)
                            senders.append(node2)
                            alreadyEdges.append((node1, node2))
                            # alreadyEdges.append((node2, node1))

    edgeFeatures = []
    for ind in range(len(senders)):
        edgeFeatures.append(np.ones(edgeFeatureCount))

    globalFeature = np.ones(globalFeatureCount)

    return {
        'n_node': sum(numNodes),
        'senders': senders,
        'receivers': receivers,
        'globals': globalFeature,
        'edges': edgeFeatures,
        'nodes': nodeFeatures,
    }


# Generates a random SGC model

# F = Original size of node features
# n_classes = Number of classes

def genModel(F, n_classes, learning_rate=0.2, l2_reg=5e-6, shouldUseBias=True):
    # Set multi-input size
    X_in = Input(shape=(F,))
    fltr_in = Input((None,), sparse=True)

    # Creates SGC layer
    output = GraphConv(n_classes,
                       activation='softmax',
                       kernel_regularizer=l2(l2_reg),
                       use_bias=shouldUseBias)([X_in, fltr_in])

    # Build model
    model = Model(inputs=[X_in, fltr_in], outputs=output)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Uses the guassian kernel to convert the input matrix into an affinity matrix
# delta = parameter for the gaussian kernel. ZIP and Doca have different ideal delta values (ZIP=5.4, DOCA=0.3)
# These ideal values were found by Kendrick

def convertToAffinityMatrixGaus(A, delta):
    two_d2 = 2 * (delta ** 2)
    val = np.exp(np.divide(-np.square(A), two_d2))

    return val


# Uses the inverse kernel to convert the input matrix into an affinity matrix
def convertToAffinityMatrixInverseKernel(A, delta):
    return delta / (A + delta)


# This function is used to generate the required data from event data to train an SGC model
# This assumes the incoming track data is only for one event
#
# @Params
# adjMatrixMode:
# ones = connects all nodes with each other with equal weight(1)
# zip = uses the zip matrix to create an affinity matrix and uses it as the adjacency matrix
# doca = uses the zip matrix to create an affinity matrix and uses it as the adjacency matrix
# identity = can collect information only from itself, i.e. there are no edges with other nodes.
# random = creates a random adjacency matrix based on the randEdgeProbability

# adjFilterKernel
# @Params
# gaussian = Use Gaussian kernel - Only applied on zip and doca
# inverse = Use Inverse kernel - Only applied on zip and doca

# featuresList
# None - creates a list of ones of specified size
# string array of column names - Uses the specified columns as the node features
# random - creates a random array

def generateSpectralDataFromTracks(tracks, featureList=None, altFeaturesSize=1, altFeaturesData=1,
                                   dataSplit=[50, 25, 25], adjMatrixMode='ones', adjFilterKernel='gaussian', delta=1,
                                   randEdgeProbability=0.5, randNodeFeatProb=0.5):
    totalNodes = len(tracks.index)

    # Creates adjacency matrix based on the specified parameter

    # Creates an totalNodes x totalNodes matrix, where each element is set to '1'. This simulates a graph where all
    # nodes are connected to each other with equal weight
    if adjMatrixMode.lower() == 'ones':
        A = np.ones((totalNodes, totalNodes))

    # Uses the ZIP matrix as the adjacency matrix. Then either the gaussian or inverse kernel is used to produce the
    # affinity matrix
    elif adjMatrixMode.lower() == 'zip':
        zipVals = tracks['zip']
        mat = []
        for v1 in zipVals:
            temp = []
            for v2 in zipVals:
                temp.append(np.abs(v1 - v2))
            mat.append(temp)

        if adjFilterKernel.lower() == 'gaussian':
            A = convertToAffinityMatrixGaus(np.array(mat), delta)
        elif adjFilterKernel.lower() == 'inverse':
            A = convertToAffinityMatrixInverseKernel(np.array(mat), delta)

    # Uses the DOCA matrix as the adjacency matrix. Then either the gaussian or inverse kernel is used to produce the
    # affinity matrix
    elif adjMatrixMode.lower() == 'doca':
        tracksDict = tracks.transpose().to_dict()

        if adjFilterKernel.lower() == 'gaussian':
            A = convertToAffinityMatrixGaus(genPOCADist(tracksDict), delta)
        elif adjFilterKernel.lower() == 'inverse':
            A = convertToAffinityMatrixInverseKernel(genPOCADist(tracksDict), delta)

    # Creates an identity matrix of size [totalNodes,totalNodes].
    # This simulates a graph where no nodes are connected to each other
    elif adjMatrixMode.lower() == 'identity':
        A = np.identity(totalNodes)

    # Creates a random matrix of size [totalNodes,totalNodes]. Each element could either be zero or one, depending on
    # the randEdgeProbability
    elif adjMatrixMode.lower() == 'random':
        A = np.random.choice([0, 1], size=(totalNodes, totalNodes), p=[1.0 - randEdgeProbability, randEdgeProbability])

    else:
        raise ValueError('The only valid arguments for adjacency matrix mode are : ones,zip,doca,identity,random')

    # Creates a nodeFeatures matrix depending on the arguments specified
    nodeFeatures = []

    # Creates a matrix of size [totalNodes, altFeaturesSize], where each element is {altFeaturesData}
    if featureList is None:
        nodeFeatures = np.full((totalNodes, altFeaturesSize), altFeaturesData)

    # Creates a random matrix of size [totalNodes, altFeaturesSize], where each element has a probability of being
    # 1 or 0 based on {randNodeFeatProb}

    elif featureList is 'random':
        nodeFeatures = np.random.choice([0, 1], size=(totalNodes, altFeaturesSize),
                                        p=[1 - randNodeFeatProb, randNodeFeatProb])

    # Creates a matrix based from the columns specified in the featuresList
    else:
        onlyFeatures = tracks[featureList].to_numpy()
        for i in range(len(onlyFeatures)):
            tempArr = []
            for vals in onlyFeatures[i]:
                try:
                    tempArr.extend(vals)
                except TypeError:
                    tempArr.append(vals)
            nodeFeatures.append(tempArr)
        nodeFeatures = np.array(nodeFeatures)

    # Gets track labels
    y_temp = tracks['track_label'].to_numpy()
    uniqueLabels = np.unique(y_temp)
    numClasses = len(uniqueLabels)

    y = []
    for val in y_temp:
        tempArr = np.zeros(numClasses)
        ind = np.where(uniqueLabels == val)
        tempArr[ind[0]] = 1
        y.append(tempArr)
    y = np.array(y)

    # This is lagacy code used to create boolean masks for splitting the data into training, testing and validation sets
    # This is ignored by the wrapper function, so could be removed in the future to simplify code
    train_mask = np.full(totalNodes, False)
    test_mask = np.full(totalNodes, False)
    val_mask = np.full(totalNodes, False)

    trainCount = int((dataSplit[0] / 100.0) * totalNodes)
    testCount = int((dataSplit[1] / 100.0) * totalNodes)

    ordering = np.random.permutation(totalNodes)
    for ind in ordering[0:trainCount]:
        train_mask[ind] = True
    for ind in ordering[trainCount:trainCount + testCount]:
        test_mask[ind] = True
    for ind in ordering[trainCount + testCount:-1]:
        val_mask[ind] = True

    return csr_matrix(A), csr_matrix(nodeFeatures), y, train_mask, test_mask, val_mask


# This is a wrapper function for the generateSpectralDataFromTracks()

# @Params
# adjMatrixMode:
# ones = connects all nodes with each other with equal weight(1)
# zip = uses the zip matrix to create an affinity matrix and uses it as the adjacency matrix
# doca = uses the zip matrix to create an affinity matrix and uses it as the adjacency matrix
# identity = can collect information only from itself, i.e. there are no edges with other nodes.
# random = creates a random adjacency matrix based on the randEdgeProbability

# adjFilterKernel
# @Params
# gaussian = Use Gaussian kernel - Only applied on zip and doca
# inverse = Use Inverse kernel - Only applied on zip and doca

# featuresList
# None - creates a list of ones of specified size
# string array of column names - Uses the specified columns as the node features
# random - creates a random array

def genDataForEvents(tracks, K=2, featuresList=None, altFeaturesSize=1, altFeaturesData=1, adjMatrixMode='ones',
                     adjFilterKernel='gaussian', delta=1, randEdgeProbability=0.5, randNodeFeatProb=0.5):
    res = {}
    uniqueEvents = tracks['eventId'].unique()

    # Loop through each event, and generate the required data structures for it
    for eventId in uniqueEvents:
        event = tracks[tracks.eventId == eventId]

        A, X, y, train_mask, test_mask, val_mask = generateSpectralDataFromTracks(event, featureList=featuresList,
                                                                                  dataSplit=[100, 0, 0],
                                                                                  adjMatrixMode=adjMatrixMode,
                                                                                  adjFilterKernel=adjFilterKernel,
                                                                                  delta=delta,
                                                                                  altFeaturesSize=altFeaturesSize,
                                                                                  altFeaturesData=altFeaturesData,
                                                                                  randEdgeProbability=randEdgeProbability,
                                                                                  randNodeFeatProb=randNodeFeatProb)

        # Prepares the Adjacency matrix by multiplying it K-1 times with itself, as this pre-computational step is
        # required for the SGC
        fltr = localpooling_filter(A).astype('f4')
        for i in range(K - 1):
            fltr = fltr.dot(fltr)
        fltr.sort_indices()

        dat = {'A': A, 'X': X.toarray(), 'y': y, 'fltr': fltr}
        res[eventId] = dat
    return res


# This is a special wrapper function for the generateSpectralDataFromTracks(), specific for generating data for the
# different K experiment

# Since the processing of K is the step that takes the longest time, this works by using the value from the
# previous step to process the data for the next step, instead of calculating from the beginning

# For example, if the matrix for K=8 has to be calculated, the adjacency matrix has to be
# raised to the power of K-1

# If data has to be processed for different K values, the above genDataForEvents()
# would naively calculate A^(K-1) for each different K

# This function aims to optimise that by sorting the kVals array in ascending order and then using the
# value from the previous step to process the data for the next step, instead of calculating from the beginning

# @Params

# kVals: array of different K values

# adjMatrixMode:
# ones = connects all nodes with each other with equal weight(1)
# zip = uses the zip matrix to create an affinity matrix and uses it as the adjacency matrix
# doca = uses the zip matrix to create an affinity matrix and uses it as the adjacency matrix
# identity = can collect information only from itself, i.e. there are no edges with other nodes.
# random = creates a random adjacency matrix based on the randEdgeProbability

# adjFilterKernel:
# gaussian = Use Gaussian kernel - Only applied on zip and doca
# inverse = Use Inverse kernel - Only applied on zip and doca

# featuresList:
# None - creates a list of ones of specified size
# string array of column names - Uses the specified columns as the node features
# random - creates a random array

def genDataForDiffK(tracks, kVals, featuresList=None, altFeaturesSize=1, altFeaturesData=1, adjMatrixMode='ones',
                    adjFilterKernel='gaussian', delta=1, randEdgeProbability=0.5, randNodeFeatProb=0.5):
    res = {}
    uniqueEvents = tracks['eventId'].unique()
    filters = {}
    # Loop through each event, and generate the required data structures for it
    for eventId in uniqueEvents:
        eventFilters = {}
        event = tracks[tracks.eventId == eventId]

        A, X, y, train_mask, test_mask, val_mask = generateSpectralDataFromTracks(event, featureList=featuresList,
                                                                                  dataSplit=[100, 0, 0],
                                                                                  adjMatrixMode=adjMatrixMode,
                                                                                  adjFilterKernel=adjFilterKernel,
                                                                                  delta=delta,
                                                                                  altFeaturesSize=altFeaturesSize,
                                                                                  altFeaturesData=altFeaturesData,
                                                                                  randEdgeProbability=randEdgeProbability,
                                                                                  randNodeFeatProb=randNodeFeatProb)

        # For the below code to work, it is imperative for kVals to be sorted in ascending order
        kVals.sort()

        # Prepares the Adjacency matrix by multiplying it K-1 times with itself, as this pre-computational step is
        # required for the SGC, and uses the aforementioned optimizations

        fltr = localpooling_filter(A).astype('f4')
        prevK = None
        for K in kVals:
            currK = K

            if prevK is None:
                for i in range(currK):
                    fltr = fltr.dot(fltr)
            else:
                for i in range(currK - prevK):
                    fltr = fltr.dot(fltr)

            fltr.sort_indices()
            prevK = currK
            eventFilters[K] = fltr

        dat = {'A': A, 'X': X.toarray(), 'y': y}
        res[eventId] = dat
        filters[eventId] = eventFilters

    finalRes = {}
    for K in kVals:
        finalRes[K] = {}
        for eventId in uniqueEvents:
            finalRes[K][eventId] = {'A': res[eventId]['A'], 'X': res[eventId]['X'], 'y': res[eventId]['y']}
            finalRes[K][eventId]['fltr'] = filters[eventId][K]

    return finalRes


# Loads events, processes them and removed global outliers, and returns as labelled tracks

# @Params
# PVFileName = The 'name' of the event file being loaded.
#             Due to the way the importer.extractData() is structured, the tracks and pvs file must both be available
#             in the specified directory.
#             It expects the particle file and tracks file to be formatted as "pv_{num}pvs.root" and
#             "trks_{num}pvs.root" respectively.
#             Therefore, the argument for PVFileName would just be {num}
#             For example, when loading the pv_100pvs.root and trks_100pvs.root files,
#             the argument for eventFile would be '100'
#
# path = location of the directory where the event files are located

# eventsToLoad = Optional parameter to choose how many events to load from the specified events file
#                     By default, it loads all events, but if you want to load only the first 10 events from the
#                     100 events file, the value for this parameter could be set to '10'

def loadAndPrepareAllEvents(path, PVFileName, eventsToLoad=None):
    dataDir = Path(path)

    sims, recTks = loadAllEvents(PVFileName, dataDir, eventsToLoad)  # Loads all events
    allGTTracks = poolAllGTEventTracks(sims)  # Creates tracks from events and combines them into one dictionary
    allGT_ZIP = genZip(allGTTracks)  # Creates zip values for all tracks
    allGTTracks_ZIP = addZipToTracks(allGTTracks, allGT_ZIP)  # Adds zip values to dictionary
    allGT_pd = pd.DataFrame(
        allGTTracks_ZIP).transpose()  # Create pandas dataframe which is used by the rest of the pipeline
    removedGB = removeGlobalOutliersByZScore(allGT_pd)  # Removes global outliers by zScore
    all_labelled = labelTracks(removedGB)  # Generates labels for tracks using Ground truth ZIP distribution

    return all_labelled


# Custom accuracy function to test tensorflow's model.evaluate()
# Target and predicted are numpy arrays of hot vectors
# Both the inputs must have the same shape
def calcAccuracy(target, predicted):
    if target.shape != predicted.shape:
        raise ValueError('Both the target and the predicted  must have the same shape')

    predicted = np.round(predicted)
    diff = target - predicted
    count = np.count_nonzero(diff == 0)
    return count / (target.shape[0] * target.shape[1])


# Custom accuracy function to test tensorflow's model.evaluate()
# Target and predicted are numpy arrays of hot vectors
# Both the inputs must have the same shape
def calcAccuracy_2(target, predicted):
    if target.shape != predicted.shape:
        raise ValueError('Both the target and the predicted must have the same shape')

    pred = []
    for row in predicted:
        ind = np.argmax(row)
        newArr = np.zeros(len(row))
        newArr[ind] = 1
        pred.append(newArr)

    pred = np.array(pred)
    diff = target - pred
    count = np.count_nonzero(diff == 0)
    return count / (target.shape[0] * target.shape[1])


def calcAccuracy_3(target, predicted):
    if target.shape != predicted.shape:
        raise ValueError('Both the target and the predicted must have the same shape')

    diff = np.abs(target - predicted)
    return 1.0 - np.sum(diff) / (target.shape[0] * target.shape[1])
