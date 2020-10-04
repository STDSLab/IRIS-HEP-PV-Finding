import numpy as np

from Code.prepareData import convertToAffinityMatrixGaus
from sklearn.metrics import pairwise_distances
import Code.fnmtf.factorize as fnmtf


def calcTotalError(error):
    total = 0
    for er in error:
        total += sorted(er.items(), key=lambda x: x[1])[0][1]
    return total


def run(X, numClusters, maxIter, saveDir='', thresh=0.9):

    mats, errors = fnmtf.main(X, filename=f"{saveDir}inputFactTest.npz", k=f'{numClusters}', iterations=maxIter)
    newErrors = {it: er for it, er in enumerate(errors)}

    U = mats[0]
    vals = [(val, it) for it, val in enumerate(U[:, 0])]
    sortedVals = sorted(vals, key=lambda x: x[0], reverse=True)

    maxVal = sortedVals[0][0]
    labels = [1 if z/maxVal >= thresh else 0 for z, it in vals]

    F = np.zeros((X.shape[0], 1))
    F[:, 0] = labels

    return F, mats[1], newErrors, len(errors)


def KRuns(zipVals, numClusters, maxIter=1000, errRate=100, stopTolerance=np.inf,
          overallReps=1, distMetric='l2', gaussPeram=5.4, saveDir='', thresh=0.9):
    results = []
    mat = pairwise_distances(zipVals, metric=distMetric)
    X = convertToAffinityMatrixGaus(mat, gaussPeram)

    for _ in range(overallReps):
        matrices = {'F': [], 'S': [], 'X': [], 'errors': []}
        input = X

        tracksRemSoFar = set()

        for count in range(numClusters):

            F, S, errors, _ = run(X=input, numClusters=1, maxIter=maxIter, saveDir=saveDir, thresh=thresh)

            matrices['S'].append(S)
            matrices['X'].append(input)
            matrices['errors'].append(errors)

            trackInds = np.nonzero(F)[0]
            tracksRemSoFar.update(trackInds)

            newF = np.zeros((len(zipVals), numClusters))
            newF[trackInds, count] = 1

            matrices['F'].append(newF)

            temp = np.nonzero(F == 0)[0]
            newInputIds = [t for t in temp if t not in tracksRemSoFar]

            if len(newInputIds) == 0:
                break

            input = np.zeros(X.shape)

            for i in newInputIds:
                for j in newInputIds:
                    input[i, j] = X[i, j]

        results.append((calcTotalError(matrices['errors']), matrices))

    return sorted(results, key=lambda x: x[0])[0][1]
