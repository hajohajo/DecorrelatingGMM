import os
import shutil
from root_pandas import read_root
from hyperOptimization import COLUMNS
import glob

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
        dataset = read_root(filePaths, columns=COLUMNS)
        dataset["eventType"] = eventTypeDict[identifier]
        listOfDatasets.append(dataset)

    return listOfDatasets[0], listOfDatasets[1:]

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