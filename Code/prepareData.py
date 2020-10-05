# Author: Alan Joshua Aneeth Jegaraj (The first few methods are authored by Kendrick Li)


from pathlib import Path
import numpy as np
import pandas as pd
import uproot
from scipy import stats
import pickle


def z2coord(cPos, cTraj, z, z0):
    return cPos + cTraj * (z - z0)


def z2xy(xyPos, xyTraj, z, z0):
    return z2coord(xyPos[0], xyTraj[0], z, z0), z2coord(xyPos[1], xyTraj[1], z, z0)


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


def genZipMatrix(tracks):
    zipVals = tracks['zip']
    mat = []
    for v1 in zipVals:
        temp = []
        for v2 in zipVals:
            temp.append(np.abs(v1 - v2))
        mat.append(temp)
    mat = np.array(mat)
    return mat


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


# removes global outliers using the zip distribution of the entire event and
# excluding tracks outside the specified IQR range
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


# removes global outliers using the zip distribution of the entire event and excluding tracks outside the
# specified zScore
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


# Labels all tracks as either core(-1)/non-core(0)/outlier(1) using ground truth zip distribution
def labelTracks(tracks, zThresh=(0.5, 1.5)):
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

            # Labels each track based on the (default) criteria:
            #   |z| <= 0.5 : core (-1)
            #   0.5 < |z| <= 1.5 : non-core (0)
            #   |z| > 1.5 : outlier (-1)

            labels = []
            for index in range(len(zScores)):

                curZScore = zScores[index]
                if np.abs(curZScore) <= zThresh[0]:
                    labels.append(-1)
                elif np.abs(curZScore) <= zThresh[1]:
                    labels.append(0)
                else:
                    labels.append(1)

            labels = np.array(labels)
            coreTrackInd = np.where(labels == -1)[0]

            if len(coreTrackInd) <= 1:
                # If no core tracks are labelled, mark all tracks as Core.
                # Since those only occurs in PVs with a low count of tracks,
                # marking them all as core would be the ideal thing to do
                labels[:] = -1

            singlePVTracks['track_label'] = labels
            newTracks = newTracks.append(singlePVTracks)
    return newTracks


def loadAllEvents(eventFile, dataDir, totalEventsToPool=None, loadSingleEvent=None, loadOnlySpecificEvents=None):
    if totalEventsToPool is None:
        totalEventsToPool = eventFile
    allSims = {}
    allTKRecs = {}
    if loadSingleEvent is None:
        if loadOnlySpecificEvents is None:
            for eventID in range(totalEventsToPool):
                sim, tkRec = importer(eventFile, dataDir).extractData(eventID)
                allSims[eventID] = sim
                allTKRecs[eventID] = tkRec
        else:
            for eventID in loadOnlySpecificEvents:
                sim, tkRec = importer(eventFile, dataDir).extractData(eventID)
                allSims[eventID] = sim
                allTKRecs[eventID] = tkRec
    else:
        eventID = loadSingleEvent
        sim, tkRec = importer(eventFile, dataDir).extractData(eventID)
        allSims[eventID] = sim
        allTKRecs[eventID] = tkRec
    return allSims, allTKRecs


# Gets a list of simulations (each simulation holds data about one event), generates tracks from them and
# combines them into one dictionary

def poolAllGTEventTracks(sims):
    tracks = {}
    lastTrackID = 0
    for eventInd, sim in sims.items():
        currTr = genGTTracks(sim, eventInd)
        for key, ind in zip(currTr.keys(), range(len(currTr))):
            tracks[ind + lastTrackID] = currTr[key]
        lastTrackID += len(currTr)
    return tracks


def loadAndPrepareAllEvents_labelled(path, PVFileName, eventsToLoad=None, loadSingleEvent=None, loadOnlySpecificEvents=None,
                            zThresh=(0.5, 1.5)):
    dataDir = Path(path)

    # Loads all events
    sims, recTks = loadAllEvents(PVFileName, dataDir, eventsToLoad, loadSingleEvent,
                                 loadOnlySpecificEvents=loadOnlySpecificEvents)
    allGTTracks = poolAllGTEventTracks(sims)  # Creates tracks from events and combines them into one dictionary
    allGT_ZIP = genZip(allGTTracks)  # Creates zip values for all tracks
    allGTTracks_ZIP = addZipToTracks(allGTTracks, allGT_ZIP)  # Adds zip values to dictionary
    allGT_pd = pd.DataFrame(
        allGTTracks_ZIP).transpose()  # Create pandas dataframe which is used by the rest of the pipeline
    # removedGB = removeGlobalOutliersByZScore(allGT_pd)  # Removes global outliers by zScore
    all_labelled = labelTracks(allGT_pd,
                               zThresh=zThresh)  # Generates labels for tracks using Ground truth ZIP distribution

    return all_labelled


# Wrapper function to load events to perform clustering
def loadAndPrepareAllEvents(path, PVFileName, eventsToLoad=None, loadSingleEvent=None):
    dataDir = Path(path)
    sims, recTks = loadAllEvents(PVFileName, dataDir, eventsToLoad, loadSingleEvent)  # Loads all events
    allGTTracks = poolAllGTEventTracks(sims)  # Creates tracks from events and combines them into one dictionary
    allGT_ZIP = genZip(allGTTracks)  # Creates zip values for all tracks
    allGTTracks_ZIP = addZipToTracks(allGTTracks, allGT_ZIP)  # Adds zip values to dictionary
    allGT_pd = pd.DataFrame(
        allGTTracks_ZIP).transpose()  # Create pandas dataframe which is used by the rest of the pipeline

    return allGT_pd


def createZipDistMatrix(tracks):
    zipVals = tracks['zip']
    mat = []
    for v1 in zipVals:
        temp = []
        for v2 in zipVals:
            temp.append(np.abs(v1 - v2))
        mat.append(temp)
    mat = np.array(mat)
    return mat


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


# Generates the S and F matrix from input matrix and the generated F Matrix
def genSFromFAndX(F, targX):
    FtF = F.transpose().dot(F)
    inter1 = (np.linalg.lstsq(FtF, F.transpose()))[0]
    S = np.linalg.lstsq(FtF.T, inter1.dot(targX).dot(F).T)[0]

    return S


def saveResults(resVals, resFile):
    with open(resFile, 'wb') as handle:
        pickle.dump(resVals, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadFile(fileName):
    with open(fileName, 'rb') as handle:
        return pickle.load(handle)


# def saveEventResultsToExcel(all, core, saveLoc):
#     writer = pd.ExcelWriter(saveLoc)
#
#     for eventId in all.keys():
#         allData = getResultsPerEvent(all[eventId])
#         coreData = getResultsPerEvent(core[eventId])
#         allData.to_excel(writer, sheet_name=f'allTracks_event={eventId}')
#         coreData.to_excel(writer, sheet_name=f'coreTracks_event={eventId}')
#
#     writer.close()