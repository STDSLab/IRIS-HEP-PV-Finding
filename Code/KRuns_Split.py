
import numpy as np
from sklearn.metrics import pairwise_distances
import Code.fnmtf.factorize as fnmtf
from Code.prepareData import convertToAffinityMatrixGaus
from Code.FNMTF import FNMTF as secondaryFNMTF


def calcTotalError(error):
    total = 0
    for er in error:
        total += sorted(er.items(), key=lambda x: x[1])[0][1]
    return total


def cluster(X, numClusters, maxIter, errRate=100, stopTolerance=np.inf, saveDir='', ignoreTracks=[],
            thresh=0.1, tksOutMinThresh=0.5):
    mats, errors = fnmtf.main(X, filename=f"{saveDir}inputFactTest.npz", k=f'{numClusters}', iterations=maxIter)
    newErrors = {it: er for it, er in enumerate(errors)}

    U = mats[0]

    vals = [(val, it) for it, val in enumerate(U[:, 0])]
    sortedVals = sorted(vals, key=lambda x: x[0], reverse=True)

    maxVal = sortedVals[0][0]
    tempVals = np.array([z / maxVal for z, it in vals])

    labels = [1 if z >= thresh else 0 for z in tempVals]
    tksOutlier = [1 if ((z < thresh) and (z >= tksOutMinThresh)) else 0 for z in tempVals]

    F = np.zeros((X.shape[0], 1))
    F[:, 0] = labels

    return F, mats[1], newErrors, len(errors), np.nonzero(tksOutlier)[0]


def KRuns_Split(zipVals, numClusters, maxIter=1000, errRate=100, stopTolerance=np.inf,
                      overallReps=1, distMetric='l2', gaussPeram=5.4, saveDir='', thresh=0.9, old_FNMTF_thresh=0.2,
                      old_FNMTF_rep_count=5, stopPoint=0.05, minTksInCluster=5, diffThresh=25, tksOutMinThresh=0.5):

    results = []
    mat = pairwise_distances(zipVals, metric=distMetric)
    X = convertToAffinityMatrixGaus(mat, gaussPeram)

    for _ in range(overallReps):

        matrices = {'F': [], 'S': [], 'X': [], 'errors': []}
        input = X
        tracksRemSoFar = set()

        iterCount = 0
        done = False

        while not done:

            F, S, errors, _, tksOutliers = cluster(X=input, numClusters=1, maxIter=maxIter, errRate=errRate,
                                                   stopTolerance=stopTolerance, saveDir=saveDir,
                                                   ignoreTracks=tracksRemSoFar, thresh=thresh,
                                                   tksOutMinThresh=tksOutMinThresh)

            trackInds = np.nonzero(F)[0]
            tempInput = np.zeros((len(trackInds), len(trackInds)))
            keys = {}
            for i, x in enumerate(trackInds):
                keys[i] = x
                for j, y in enumerate(trackInds):
                    tempInput[i, j] = X[x, y]

            finalErrors = []
            old_fnmtf_runRes = []

            # To make sure FNMTF is not tried when only one track is detected
            if len(trackInds) > 1:
                for run in range(old_FNMTF_rep_count):
                    vals = secondaryFNMTF(tempInput, 2, maxIter=1000, errRate=1, stopTolerance=10)
                    finalErrors.append(list(vals[2].values())[-1])
                    old_fnmtf_runRes.append(vals)
                old_FNMTF_res = old_fnmtf_runRes[np.argmin(finalErrors)]

                newF1TrackIds = np.nonzero(old_FNMTF_res[0][:, 0])[0]
                newF2TrackIds = np.nonzero(old_FNMTF_res[0][:, 1])[0]

                numTracks1 = len(newF1TrackIds)
                numTracks2 = len(newF2TrackIds)

            else:
                numTracks1 = 0
                numTracks2 = 0

            if (numTracks1 / len(trackInds)) >= old_FNMTF_thresh and (numTracks2 / len(trackInds)) >= old_FNMTF_thresh:

                if numTracks1 >= minTksInCluster:
                    newF1 = np.zeros(len(zipVals))
                    for id in newF1TrackIds:
                        newF1[keys[id]] = 1
                    matrices['F'].append(newF1)
                    matrices['S'].append(S)
                    matrices['X'].append(input)
                    matrices['errors'].append(errors)
                    iterCount += 1

                if numTracks2 >= minTksInCluster:
                    newF2 = np.zeros(len(zipVals))
                    for id in newF2TrackIds:
                        newF2[keys[id]] = 1
                    matrices['F'].append(newF2)
                    matrices['S'].append(S)
                    matrices['X'].append(input)
                    matrices['errors'].append(errors)
                    iterCount += 1

            # Not overlapping
            else:
                if len(trackInds) >= minTksInCluster:
                    matrices['S'].append(S)
                    matrices['X'].append(input)
                    matrices['errors'].append(errors)

                    newF = np.zeros(len(zipVals))
                    newF[trackInds] = 1
                    matrices['F'].append(newF)

                    iterCount += 1

            tracksRemSoFar.update(trackInds)
            tracksRemSoFar.update(tksOutliers)

            temp = np.nonzero(F == 0)[0]
            newInputIds = [t for t in temp if t not in tracksRemSoFar]

            currClCount = 0
            clEndInd = len(newInputIds) - 1
            extraNewInputIds = []
            numTracksNotNoise = 0
            ogNewInputLen = len(newInputIds)

            for i in range(len(newInputIds) - 1, 1, -1):
                currClCount += 1
                currDiff = newInputIds[i] - newInputIds[i - 1]

                if currDiff >= diffThresh:

                    if currClCount >= minTksInCluster:
                        extraNewInputIds.extend(newInputIds[i: clEndInd + 1])
                        numTracksNotNoise += currClCount

                    clEndInd = i
                    currClCount = 0

            if currClCount >= minTksInCluster:
                extraNewInputIds.extend(newInputIds[0: clEndInd + 1])
                numTracksNotNoise += currClCount

            newInputIds = extraNewInputIds
            diffs = []
            for i in range(len(newInputIds) - 1, 1, -1):
                diffs.append(newInputIds[i] - newInputIds[i - 1])

            diffs = np.array(diffs)
            inverse_variance = -1

            if len(diffs) > 0:
                inverse_variance = np.count_nonzero(diffs == 1) / ogNewInputLen

            # print(
            #     f'Num clusters found : {len(matrices["F"])}  numClusters: {numClusters} inv_var: {inverse_variance} numTks_left={len(newInputIds) / len(zipVals)}')

            if len(newInputIds) < minTksInCluster or inverse_variance <= stopPoint:
                break

            input = np.zeros(X.shape)
            for i in newInputIds:
                for j in newInputIds:
                    input[i, j] = X[i, j]

        results.append((calcTotalError(matrices['errors']), matrices))

    return sorted(results, key=lambda x: x[0])[0][1]
