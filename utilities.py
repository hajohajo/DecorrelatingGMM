import os
import shutil
from root_pandas import read_root
from hyperOptimization import COLUMNS_, PTMIN, PTMAX, PTBINS
import glob
import numpy as np
from sklearn.utils import compute_sample_weight


def getClassifierSampleWeights(dataframe):
    weights = compute_sample_weight('balanced', dataframe.loc[:, "eventType"])

    return weights

def getAdversarySampleWeights(dataframe):
    binning = np.linspace(PTMIN, PTMAX, PTBINS+1)
    digitizedSamples = np.digitize(np.clip(dataframe['TransverseMass'].values, PTMIN, PTMAX-1.0),
                                   bins=binning, right=False).astype(np.float32)
    weights = compute_sample_weight('balanced', digitizedSamples)
    weights[dataframe.eventType == 0] = 0

    return weights

def clipDataFrameToQuantiles(dataframe, lowerQuantile=0.2, upperQuantile=0.8):
    if(len(dataframe.shape) > 1):
        returnFrame = dataframe.clip(dataframe.quantile(lowerQuantile), dataframe.quantile(upperQuantile), axis=1)
        return returnFrame
    returnFrame = dataframe.clip(dataframe.quantile(lowerQuantile), dataframe.quantile(upperQuantile))
    return returnFrame

class quantileClipper():
    def __init__(self, lowerQuantile=0.1, upperQuantile=0.9):
        self.lowerQuantile = lowerQuantile
        self.upperQuantile = upperQuantile
        self.fitCalled = False

    def fit(self, dataframe):
        self.fitCalled = True
        self.lowerThresholds = dataframe.quantile(self.lowerQuantile)
        self.upperThresholds = dataframe.quantile(self.upperQuantile)

    def clip(self, dataframe):
        newDataframe = dataframe.clip(self.lowerThresholds, self.upperThresholds, axis=1)
        return newDataframe


eventTypeDict = {
    "ChargedHiggs_" : 0,
    "TT_" : 1,
    "DYJets" : 2,
    "QCD_" : 3,
    "ST_" : 4,
    "WJets" : 5,
    "WW" : 6,
    "WZ" : 6,
    "ZZ" : 6
}

invertedEventTypeDict = {
    0 : "Signal",
    1 : "TT",
    2 : "DY",
    3 : "QCD",
    4 : "ST",
    5 : "WJets",
    6 : "Diboson"
}

def readDatasetsToDataframes(pathToFolder):
    listOfDatasets = []
    identifiers = ["ChargedHiggs_", "TT_", "DYJets", "QCD_", "ST_", "WJets", "WW", "WZ", "ZZ"]
    for identifier in identifiers:
        filePaths = glob.glob(pathToFolder + identifier+"*.root")
        dataset = read_root(filePaths, columns=COLUMNS_)
        dataset["eventType"] = eventTypeDict[identifier]
        listOfDatasets.append(dataset)

    numberOfSignalEvents = listOfDatasets[0].shape[0]
    numberOfBackgroundEvents = np.sum([x.shape[0] for x in listOfDatasets[1:]])
    if(numberOfSignalEvents>numberOfBackgroundEvents):
        listOfDatasets[0] = listOfDatasets[0].sample(n=numberOfBackgroundEvents)

    dataframe = listOfDatasets[0].append(listOfDatasets[1:])

    return dataframe

def createDirectories():
    directories = ["plots", "logs"]

    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)

    if(not os.path.exists("models")):
        os.makedirs("models")
