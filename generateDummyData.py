import numpy as np
import pandas as pd
from numpy.random import randn, rand

eventTypeDict = {"ChargedHiggs" : 0,
                 "TT" : 1,
                 "WJets" : 2,
                 "QCD" : 3}

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def makeDistributions(nSamples, mass, eventType):
    isSignal = (eventType is "ChargedHiggs")
    metFractions = rand(nSamples)
    tauPt = np.clip(np.clip((1-np.exp(-0.01*mass)), 0.0, 1.0) * mass + randn(mass.shape[0]), 5.0, 200.0)
    MET = mass-tauPt
    if isSignal:
        ldgTrkPtFrac = sigmoid(randn(nSamples)+2.0) #sigmoid(4 * rand(nSamples))
        ra = rand(nSamples)
        deltaPhiTauMet = (2 * ra * ra - 1) * np.pi
#        deltaPhiTauMet = (2 * rand(nSamples) - 1) * np.pi

    else:
        ldgTrkPtFrac = sigmoid(8 * rand(nSamples) - 4.0)
        deltaPhiTauMet = (2*rand(nSamples)-1) * np.pi
    bjetPt = np.clip(np.random.exponential(60.0, nSamples), 5.0, 200.0)
    deltaPhiTauMet = rand(nSamples) * np.pi
    deltaPhiTauBjet = rand(nSamples) * np.pi
    deltaPhiBjetMet = rand(nSamples) * np.pi
    if isSignal:
        target = np.ones(nSamples)
    else:
        target = np.zeros(nSamples)

    eventType = np.ones(nSamples) * eventTypeDict[eventType]

    return np.column_stack((MET, tauPt, ldgTrkPtFrac, deltaPhiTauMet, deltaPhiTauBjet, bjetPt, deltaPhiBjetMet, mass, target, eventType))


def generateSamples(nSamples=10000, isSignal=True):
    columns = ["MET", "tauPt", "ldgTrkPtFrac", "deltaPhiTauMet", "deltaPhiTauBjet", "bjetPt", "deltaPhiBjetMet", "TransverseMass", "target", "EventType"]
    signalMassPoints = [100] #[30, 60, 80, 100, 120, 180] #, 300, 400, 500]
    backgroundMassPoints = [80, 90, 173]

    dataframe = pd.DataFrame(columns=columns)

    if isSignal:
        for massPoint in signalMassPoints:
            mass = 10*randn(nSamples) + massPoint
            columnValues = makeDistributions(nSamples, mass, "ChargedHiggs")
            df = pd.DataFrame(data=columnValues,
                              columns=columns)

            dataframe = dataframe.append(df)


        return dataframe

    else:
        # mass = 50*randn(nSamples) + 80
        mass = 200*rand(nSamples)
        columnValues = makeDistributions(nSamples, mass, "WJets")

        df = pd.DataFrame(data=columnValues,
                          columns=columns)

        dataframe = dataframe.append(df)

        # mass = 50*randn(nSamples) + 173
        # columnValues = makeDistributions(nSamples, mass, "TT")
        #
        # df = pd.DataFrame(data=columnValues,
        #                   columns=columns)
        #
        # dataframe = dataframe.append(df)
        #
        #
        # mass = np.random.exponential(80.0, nSamples)
        # columnValues = makeDistributions(nSamples, mass, "QCD")
        # df = pd.DataFrame(data=columnValues,
        #                   columns=columns)
        # dataframe = dataframe.append(df)
        return dataframe