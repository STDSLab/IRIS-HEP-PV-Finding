# This is an implementation of Fast Non-negative Matrix Trifactorization, specialized for symmetric matrices
# This code is based on Kendrick Li's MatLab implementation of the same

# Wang, Hua, et al. "Fast nonnegative matrix tri-factorization for
# %       large-scale data co-clustering." Twenty-Second International Joint
# %       Conference on Artificial Intelligence. 2011

# Author: Alan Joshua Aneeth Jegaraj


from PV_methods import *


# inputs:
# n = sampleSize
# k = numClusters
def genF(n, k):
    F = np.zeros((n, k))
    randSec = np.random.randint(0, k, n - k)
    objMem = list(range(k))
    objMem.extend(randSec)
    objMem = np.array(objMem)
    objMem = np.random.permutation(objMem)

    for i in range(n):
        F[i, objMem[i]] = 1

    return F


def FNMTF(X, numClusters, maxIter, errRate, stopTolerance = np.inf):
    if X.shape[0] != X.shape[1]:
        raise ValueError('Matrix must be symmetric')

    n = X.shape[0]
    F = genF(n, numClusters)

    bestF = np.zeros(F.shape)
    bestErr = np.inf
    errors = {}

    tolCount = 0
    iterationsCompleted = 0

    for itCount in range(maxIter):
        FtF = F.transpose().dot(F)
        inter1 = (np.linalg.lstsq(FtF, F.transpose()))[0]
        # S = np.linalg.solve(FtF.conj().T, inter1.dot(X).dot(F).conj().T)\
        #     .conj().T
        S = np.linalg.lstsq(FtF.T, inter1.dot(X).dot(F).T)[0]

        if itCount % errRate == 0:
            recX = F.dot(S).dot(F.transpose())
            currErr = np.linalg.norm(X - recX, ord='fro')**2
            errors[itCount] = currErr

            if bestErr > currErr:
                bestF = F
                bestErr = currErr
            else:
                tolCount += 1

        if tolCount >= stopTolerance:
            iterationsCompleted = itCount
            break

        SF = S.dot(F.transpose())
        F[:] = 0

        for i in np.random.permutation(n):
            minInd = np.argmin(
                np.sum((SF - X[i, :]) ** 2, axis=1))
            F[i, minInd] = 1

        for i in range(numClusters):
            memNum = np.sum(F, axis=0)
            if memNum[i] == 0:
                maxInd = np.argmax(memNum)
                nonZ = np.where(np.any(F[:, maxInd] > 0))[0]
                nonZ = nonZ[np.random.permutation(len(nonZ))]
                F[nonZ[0], :] = 0
                F[nonZ[0], i] = 1

    FtF = F.transpose().dot(F)
    inter1 = (np.linalg.lstsq(FtF, F.transpose()))[0]
    # S = np.linalg.solve(FtF.conj().T, inter1.dot(X).dot(F).conj().T).conj().T
    S = np.linalg.lstsq(FtF.T, inter1.dot(X).dot(F).T)[0]

    recX = F.dot(S).dot(F.transpose())
    currErr = np.linalg.norm(X - recX, ord='fro') ** 2

    if bestErr < currErr:
        F = bestF

        FtF = F.transpose().dot(F)
        inter1 = (np.linalg.lstsq(FtF, F.transpose()))[0]
        # S = np.linalg.solve(FtF.conj().T, inter1.dot(X).dot(F).conj().T).conj().T
        S = np.linalg.lstsq(FtF.T, inter1.dot(X).dot(F).T)[0]

    return F, S, errors, iterationsCompleted


