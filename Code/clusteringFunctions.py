# Author: Alan Joshua Aneeth Jegaraj (aneethaj@mail.uc.edu)


import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import cluster as cl

from Code.FNMTF import FNMTF
from Code.KRuns import KRuns
from Code.KRuns_Split import KRuns_Split
from Code.prepareData import createZipDistMatrix, convertToAffinityMatrixGaus, convertToAffinityMatrixInverseKernel, \
    genSFromFAndX, getPOCABetween2Tracks


# Helper function used by clustering methods
# List of tuples, where the first element of each tuple contains the cluster ID and the
# second element contains the track


def addClusterKeyToTracks(pairs):
    tracks = {}
    for pair in pairs:
        tracks[int(pair[1].name)] = pair[1].append(pd.Series([pair[0]], name='ClusterID'), ignore_index=False)
    return pd.DataFrame(tracks).rename(index={0: 'Cluster_id'})


# Generates the F matrix for clustering algorithms that do not create their own F matrix
def createClusterMatrixFromTracks(tracks):
    clusterIds = tracks['Cluster_id'].to_numpy()
    trackCount = len(clusterIds)
    clusterCount = len(tracks['Cluster_id'].unique())

    mat = np.zeros((trackCount, clusterCount))
    for i, cl in enumerate(clusterIds):
        mat[i, cl] = 1
    return mat


# The below clustering functions have some unnecessary parameters to be consistent with lower and higher level
# functions passing and needing certain values in some cases.
# TODO: This needs to be refactored into **kwargs

def cluster_kmeans(tracks, numClusters, input_X=None, maxIter=1000, errorRate=100, tolCount=np.inf,
                   factorRepeatCount=10, FNMTF_noClChange=False, kernel='cosine', distMetric='euclidean',
                   applyKernel=True, kerParam=5.4, **kwargs):
    tracks = tracks.transpose()
    vals = [[v] for v in tracks.loc['zip']]
    centroids, y_km, _ = cl.k_means(vals, init='k-means++', n_clusters=numClusters, n_init=factorRepeatCount,
                                    max_iter=maxIter)
    pairs = [(clID, tracks[trackInd]) for clID, trackInd in zip(y_km, tracks)]
    output = addClusterKeyToTracks(pairs).transpose()
    return output, {'F': createClusterMatrixFromTracks(output)}, None  # Returning None for API consistency


# Clusters tracks using Hierarchial Agglomerative Clustering.

# Returns pandas dataframe with a new "Cluster_id" column which specifies the cluster
# to which the specific track belongs

# Requires the number of the clusters to be found as input.
# FNMTF_noClChange,errorRate=100, tolCount=np.inf,input_X, maxIter, factorRepeatCount are not used.
# They are here for the sake of API consistency

def cluster_HAC(tracks, numClusters, input_X=None, maxIter=1000, errorRate=100, tolCount=np.inf,
                factorRepeatCount=10, FNMTF_noClChange=False, applyKernel=True, kerParam=5.4, kernel='gaussian',
                distMetric='euclidean', **kwargs):
    if input_X is None:
        mat = createZipDistMatrix(tracks)
        if applyKernel:
            X = convertToAffinityMatrixGaus(mat, 5.4)
        else:
            X = mat
    else:
        mat = input_X
        X = input_X

    tracks = tracks.transpose()

    HAC = sk.cluster.AgglomerativeClustering(n_clusters=numClusters).fit(X)
    pairsHAC = [(clID, tracks[trackInd]) for clID, trackInd in zip(HAC.labels_, tracks)]
    output = addClusterKeyToTracks(pairsHAC).transpose()

    return output, {'F': createClusterMatrixFromTracks(output), 'X': mat,
                    'XGauss': X}, None  # Returning None for API consistency


# Performs clustering using FNMTF
def cluster_FNMTF(tracks, numClusters, input_X=None, maxIter=1000, errorRate=100, tolCount=np.inf,
                  repeatTimes=10, applyKernel=True, kernel='gaussian',
                  distMetric='euclidean', kerParam=5.4, **kwargs):
    from sklearn.metrics import pairwise_distances
    from sklearn.metrics.pairwise import pairwise_kernels

    if input_X is None:

        zipVals = tracks['zip'].to_numpy().reshape(-1, 1)
        mat = pairwise_distances(zipVals, metric=distMetric)
        if applyKernel:
            if kernel.lower() == 'gaussian':
                X = convertToAffinityMatrixGaus(mat, kerParam)
            elif kernel.lower() == 'inv':
                X = convertToAffinityMatrixInverseKernel(mat, kerParam)
            else:
                if kerParam is not None:
                    X = pairwise_kernels(zipVals, zipVals, metric=kernel, gamma=kerParam, filter_params=True)
                else:
                    X = pairwise_kernels(zipVals, zipVals, metric=kernel)
        else:
            X = mat
    else:
        mat = input_X
        X = input_X

    finalErrors = []
    results = []
    for run in range(repeatTimes):
        vals = FNMTF(X, numClusters, maxIter=maxIter, errRate=errorRate, stopTolerance=tolCount)
        finalErrors.append(list(vals[2].values())[-1])
        results.append(vals)

    bestRes = results[np.argmin(finalErrors)]
    F = bestRes[0]
    S = bestRes[1]
    errors = bestRes[2]

    labels = np.argmax(F, axis=1)

    tracks = tracks.transpose()
    pairs = [(clID, tracks[tracksInd]) for clID, tracksInd in zip(labels, tracks)]
    sortedList = sorted(pairs, key=lambda x: x[0])
    return addClusterKeyToTracks(sortedList).transpose(), {'X': mat, 'XGauss': X, 'F': F, 'S': S,
                                                           'RecX': (F.dot(S.dot(F.transpose())))}, errors


# KRuns Clustering
def cluster_kRuns(tracks, numClusters, input_X=None, maxIter=1000, errorRate=100, tolCount=np.inf,
                  repeatTimes=10, FNMTF_noClChange=False, applyKernel=True, kernel='gaussian',
                  distMetric='euclidean', kerParam=5.4, **kwargs):
    thresh = 0.8

    for key, val in kwargs.items():
        if key == 'thresh':
            thresh = val

    zipVals = tracks['zip'].to_numpy().reshape(-1, 1)
    matrices = KRuns(zipVals, numClusters, maxIter=maxIter, overallReps=repeatTimes, thresh=thresh)

    finalF = np.zeros((len(zipVals), numClusters))
    for index, F in enumerate(matrices['F']):
        finalF[:, index] = F[:, index]

    S = genSFromFAndX(finalF, matrices['X'][0])
    errors = matrices['errors'][-1]
    labels = np.argmax(finalF, axis=1)

    tracks = tracks.transpose()
    pairs = [(clID, tracks[tracksInd]) for clID, tracksInd in zip(labels, tracks)]
    sortedList = sorted(pairs, key=lambda x: x[0])
    return addClusterKeyToTracks(sortedList).transpose(), {'X': None, 'XGauss': matrices['X'][0], 'F': finalF, 'S': S,
                                                           'RecX': None}, errors


# KRuns+Split clustering
def cluster_kRuns_Split(tracks, numClusters, input_X=None, maxIter=1000, errorRate=100, tolCount=np.inf,
                        repeatTimes=10, FNMTF_noClChange=False, applyKernel=True, kernel='gaussian',
                        distMetric='euclidean', kerParam=5.4, **kwargs):
    thresh = 0.8
    old_FNMTF_thresh = 0.2
    old_FNMTF_rep_count = 5
    stopPoint = 0.05
    minTksInCluster = 5
    diffThresh = 25
    tksOutMinThresh = 0.5

    for key, val in kwargs.items():
        if key == 'thresh':
            thresh = val
        if key == 'old_FNMTF_thresh':
            old_FNMTF_thresh = val
        if key == 'old_FNMTF_rep_count':
            old_FNMTF_rep_count = val
        if key == 'stopPoint':
            stopPoint = val
        if key == 'minTksInCluster':
            minTksInCluster = val
        if key == 'diffThresh':
            diffThresh = val
        if key == 'tksOutMinThresh':
            tksOutMinThresh = val

    zipVals = tracks['zip'].to_numpy().reshape(-1, 1)
    matrices = KRuns_Split(zipVals, numClusters, maxIter=maxIter,
                           overallReps=repeatTimes, thresh=thresh, old_FNMTF_thresh=old_FNMTF_thresh,
                           stopPoint=stopPoint, old_FNMTF_rep_count=old_FNMTF_rep_count,
                           minTksInCluster=minTksInCluster, diffThresh=diffThresh,
                           tksOutMinThresh=tksOutMinThresh)

    finalF = np.zeros((len(zipVals), len(matrices['F'])))
    for index, F in enumerate(matrices['F']):
        finalF[:, index] = F

    S = genSFromFAndX(finalF, matrices['X'][0])
    errors = matrices['errors'][-1]
    labels = np.argmax(finalF, axis=1)

    tracks = tracks.transpose()
    pairs = [(clID, tracks[tracksInd]) for clID, tracksInd in zip(labels, tracks)]
    sortedList = sorted(pairs, key=lambda x: x[0])
    return addClusterKeyToTracks(sortedList).transpose(), {'X': None, 'XGauss': matrices['X'][0], 'F': finalF, 'S': S,
                                                           'RecX': None}, errors


# Calculate centroids for ground truth clusters
def calcGTCentroids(tracks):
    gtIds = tracks['gt'].unique()
    gt_centroids = {}
    for gtID in gtIds:
        gtTracks = tracks[tracks['gt'] == gtID]
        firstTrack = gtTracks.iloc[0]
        pv = [firstTrack['pv']['x'], firstTrack['pv']['y'], firstTrack['pv']['z']]
        gt_centroid = pv
        gt_centroids[gtID] = np.array(gt_centroid)

    return gt_centroids


# Calculate centroids for reconstructed clusters, along with position error
def calcFoundCentroids(tracks):
    clusterIDs = tracks['Cluster_id'].unique()

    beamLine = {'x': [0, 0], 'y': [0, 0], 'z': [-10000, 10000]}

    # Calculate centroids for 'found' clusters
    found_centroids = {}
    centroid_error = {}
    for clID in clusterIDs:
        try:
            clTracks = tracks[tracks['Cluster_id'] == clID]
            pocas = {}
            for index, row in clTracks.iterrows():
                # getPOCABetween2Tracks() return 2 points, one point on the track's path and the other on the beamline.
                # Here, only the point on the beamline is selected
                tempTracks = {0: row, 1: beamLine}
                pocas[index] = getPOCABetween2Tracks(tempTracks, 0, 1)[1]

            centroid = np.array([float(0), float(0), float(0)])
            for key in pocas.keys():
                centroid += np.array(pocas[key])
            centroid = centroid / len(pocas.keys())
            found_centroids[clID] = centroid

            centroid_error[clID] = (clTracks['zip'].max() - clTracks['zip'].min()) / \
                                   (2 * np.sqrt(len(clTracks['zip'].index)))

        except:
            print(clID)
            print(clTracks)
            raise ValueError('Error while calculating Found Centroids')

    return found_centroids, centroid_error


# Calculates and returns centroids for ground truth clusters and reconstructed clusters,
# along with position error in reconstructed clusters
def calculateCentroidForFoundAndGTClusters(tracks):
    found_centroids, found_centroid_error = calcFoundCentroids(tracks)
    gt_centroids = calcGTCentroids(tracks)

    return found_centroids, gt_centroids, found_centroid_error


def evaluate_default(found_clusters, gt_clusters, tracks, distMetric=500e-3):
    from scipy.optimize import linear_sum_assignment

    results = {'pv': {}, 'cluster': {}}

    pvCentroids = {gt: gt_clusters[gt] for gt in gt_clusters.keys()}
    foundCentroids = {cl: found_clusters[cl] for cl in found_clusters.keys()}

    distMat = np.zeros((len(pvCentroids), len(found_clusters)))
    trackMat = np.zeros((len(pvCentroids), len(found_clusters)))
    costMat = np.zeros((len(pvCentroids), len(found_clusters)))

    for i, pv in enumerate(pvCentroids.keys()):
        results['pv'][pv] = {}
        gtTracks = tracks[tracks['gt'] == pv]
        gtTrackIndex = set(gtTracks.index.values)
        results['pv'][pv]['totalTracksInPV'] = len(gtTrackIndex)
        results['pv'][pv]['trackIds'] = list(gtTrackIndex)
        results['pv'][pv]['found'] = {'clId': -1, 'tracksFound': 0, 'dist': np.inf}
        results['pv'][pv]['loc'] = pvCentroids[pv]

        for j, cl in enumerate(foundCentroids.keys()):
            results['cluster'][cl] = {}
            clusterTracks = tracks[tracks['Cluster_id'] == cl]
            clTrackIndex = set(clusterTracks.index.values)
            results['cluster'][cl]['totalTracksInCluster'] = len(clTrackIndex)
            results['cluster'][cl]['trackIds'] = list(clTrackIndex)
            results['cluster'][cl]['loc'] = foundCentroids[cl]
            results['cluster'][cl]['found'] = -1

            if results['cluster'][cl][
                'loc'] is not None:  # Special case most likely to happen when cl count does not match with pv count
                dist = np.linalg.norm(
                    np.array(results['pv'][pv]['loc'][2]) - np.array(results['cluster'][cl]['loc'][2]))
            else:
                dist = np.inf
            distMat[i, j] = dist

            tracksInCommon = len(list(clTrackIndex.intersection(gtTrackIndex)))
            trackMat[i, j] = tracksInCommon

            if dist <= distMetric:
                #     cost = ((1 - (tracksInCommon / len(gtTrackIndex))) + (dist / distMetric)) / 2
                costMat[i, j] = tracksInCommon

    rowInd, colInd = linear_sum_assignment(costMat, maximize=True)

    pvVals = list(pvCentroids.keys())
    clVals = list(foundCentroids.keys())
    for index in range(len(pvCentroids)):

        if not (index >= len(rowInd) or index >= len(colInd)):
            pvIndex = rowInd[index]
            clIndex = colInd[index]
            cost = costMat[pvIndex, clIndex]
        else:
            cost = 0

        if cost != 0:
            pvInd = pvVals[pvIndex]
            clInd = clVals[clIndex]
            results['cluster'][clInd]['found'] = pvInd
            results['pv'][pvInd]['found'] = {'clId': clInd, 'tracksFound': trackMat[pvIndex, clIndex],
                                             'dist': distMat[pvIndex, clIndex]}
        else:
            results['cluster'][cl]['found'] = -1

    return results


def evaluate_other(found_clusters, gt_clusters, tracks, found_centroid_error, scale=5):
    results = {'pv': {}, 'cluster': {}}
    pvCentroids = {gt: gt_clusters[gt] for gt in gt_clusters.keys()}
    foundCentroids = {cl: found_clusters[cl] for cl in found_clusters.keys()}

    # Initialize cluster results data
    for cl, clLoc in foundCentroids.items():
        results['cluster'][cl] = {}
        clusterTracks = tracks[tracks['Cluster_id'] == cl]
        clTrackIndex = set(clusterTracks.index.values)
        results['cluster'][cl]['totalTracksInCluster'] = len(clTrackIndex)
        results['cluster'][cl]['trackIds'] = list(clTrackIndex)
        results['cluster'][cl]['loc'] = foundCentroids[cl]
        results['cluster'][cl]['found'] = -1

    for pv, pvLoc in gt_clusters.items():
        results['pv'][pv] = {}
        gtTracks = tracks[tracks['gt'] == pv]
        gtTrackIndex = set(gtTracks.index.values)
        results['pv'][pv]['totalTracksInPV'] = len(gtTrackIndex)
        results['pv'][pv]['trackIds'] = list(gtTrackIndex)
        results['pv'][pv]['found'] = {'clId': -1, 'tracksFound': 0, 'dist': np.inf}
        results['pv'][pv]['loc'] = pvCentroids[pv]

        for cl, clLoc in foundCentroids.items():

            if ((clLoc[2] - scale * found_centroid_error[cl]) <= pvLoc[2]) and (
                    pvLoc[2] <= (clLoc[2] + scale * found_centroid_error[cl])):
                clusterTracks = tracks[tracks['Cluster_id'] == cl]
                clTrackIndex = set(clusterTracks.index.values)
                tracksInCommon = len(list(clTrackIndex.intersection(gtTrackIndex)))

                if 'found' in results['pv'][pv]:
                    if results['pv'][pv]['found']['dist'] > (np.linalg.norm(pvLoc[2] - clLoc[2])):
                        results['pv'][pv]['found'] = {'clId': cl, 'tracksFound': tracksInCommon,
                                                      'dist': np.linalg.norm(pvLoc[2] - clLoc[2])}
                else:
                    results['pv'][pv]['found'] = {'clId': cl, 'tracksFound': tracksInCommon,
                                                  'dist': np.linalg.norm(pvLoc[2] - clLoc[2])}

                results['cluster'][cl]['found'] = pv

    return results


# Evaluates clustering, and returns results
# EvalMode=None uses the evaluation criteria which were used throughout our presentation
# EvalMode != None uses the evaluation criteria used in this paper
# Kucharczyk et al. 2014, Primary Vertex Reconstruction at LHCb

def evaluate(found_clusters, gt_clusters, tracks, found_centroid_error, distMetric=500e-3,
             evalMode=None):
    if evalMode is None:
        return evaluate_default(found_clusters, gt_clusters, tracks, distMetric=distMetric)
    else:
        return evaluate_other(found_clusters, gt_clusters, tracks, found_centroid_error)


# Wrapper function used to perform clustering on multiple events and return evaluation results

# EvalMode=None uses the evaluation criteria which were used throughout our presentation
# EvalMode != None uses the evaluation criteria used in this paper
# Kucharczyk et al. 2014, Primary Vertex Reconstruction at LHCb

def clusterAndEvaluate(tracks, clusteringFunc, maxIter=1000,
                       tolCount=2, repeatTimes=1,
                       FNMTF_noClChange=False, applyKernel=True, kernel='linear',
                       distMetric='euclidean', kerParam=5.4, extraMeta='', errorRate=100,
                       **kwargs):
    evalMode = None
    debug = False

    for key, val in kwargs.items():
        if key == 'evalMode':
            evalMode = val
        if key == 'debug':
            debug = val

    eventIds = tracks['eventId'].unique()
    finalRes = {'meta_data': {'clusteringFunc': clusteringFunc.__name__, 'maxIter': maxIter,
                              'tolCount': tolCount, 'repeatTimes': repeatTimes,
                              'FNMTF_noClChange': FNMTF_noClChange,
                              'applyKernel': applyKernel, 'kernel': kernel, 'distMetric': distMetric,
                              'kerParam': kerParam, 'extraMeta': extraMeta, 'kwargs': kwargs},
                'events': {id: {'data': None, 'matrices': None, 'error': None} for id in eventIds}}

    for eventId in eventIds:
        if debug:
            print(f'Calculating for event: {eventId}')

        eventTracks = tracks[tracks['eventId'] == eventId]
        totalNumGTPVs = len(eventTracks['gt'].unique())

        clusteredTracks, matrices, errors = clusteringFunc(eventTracks, totalNumGTPVs, maxIter=maxIter,
                                                           tolCount=tolCount,
                                                           repeatTimes=repeatTimes,
                                                           FNMTF_noClChange=FNMTF_noClChange,
                                                           applyKernel=applyKernel, kernel=kernel,
                                                           distMetric=distMetric, kerParam=kerParam,
                                                           errorRate=errorRate, **kwargs)

        found_centroids, gt_centroids, found_centroid_error = calculateCentroidForFoundAndGTClusters(
            clusteredTracks)

        res = evaluate(found_centroids, gt_centroids, clusteredTracks, found_centroid_error,
                       evalMode=evalMode)
        finalRes['events'][eventId]['data'] = res
        finalRes['events'][eventId]['error'] = errors

        finalRes['events'][eventId]['matrices'] = matrices

    return finalRes
