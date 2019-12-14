import os
import shutil
from root_pandas import read_root
from hyperOptimization import COLUMNS
import glob

eventTypeDict = {
    "ChargedHiggs_" : 1,
    "TT_" : 2,
    "DYJets" : 3,
    "QCD_" : 4,
    "ST_" : 5,
    "WJets" : 6,
    "WW" : 7,
    "WZ" : 7,
    "ZZ" : 7
}

invertedEventTypeDict = {
    1 : "Charged",
    2 : "TT",
    3 : "DY",
    4 : "QCD",
    5 : "ST",
    6 : "WJets",
    7 : "Diboson"
}

def readDatasetsToDataframes(pathToFolder):
    listOfDatasets = []
    identifiers = ["ChargedHiggs_", "TT_", "DYJets", "QCD_", "ST_", "WJets", "WW", "WZ", "ZZ"]
    for identifier in identifiers:
        filePaths = glob.glob(pathToFolder + identifier+"*.root")
        dataset = read_root(filePaths, columns=COLUMNS)
        dataset["eventType"] = eventTypeDict[identifier]
        listOfDatasets.append(dataset)

    return listOfDatasets

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