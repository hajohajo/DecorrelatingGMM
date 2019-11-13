import numpy as np
from math import ceil, floor
import matplotlib.pyplot as plt

xBinning = {"tauPt" : np.linspace(0.0, 500.0, 100),
            "MET" : np.linspace(0.0, 500.0, 100),
            "bjetPt" : np.linspace(0.0, 500.0, 100),
            "ldgTrkPtFrac" : np.linspace(0.0, 1.0, 20),
            "deltaPhiTauMet" : np.linspace(0.0, np.pi, 30),
            "deltaPhiTauBjet" : np.linspace(0.0, np.pi, 30),
            "deltaPhiBjetMet" : np.linspace(0.0, np.pi, 30),
            "TransverseMass" : np.linspace(0.0, 600.0, 100)}

xLabel = {"tauPt" : r"Tau p$\_T$",
            "MET" : r"E$_{T, miss}$",
            "bjetPt" : r"B-jet p$_T$",
            "ldgTrkPtFrac" : r"Leading charged track p$_T$",
            "deltaPhiTauMet" : r"$\Delta\Phi_{\tau, MET}$",
            "deltaPhiTauBjet" : r"\Delta\Phi_{\tau, MET}$",
            "deltaPhiBjetMet" : r"\Delta\Phi_{b jet, MET}$",
            "TransverseMass" : r"m$_T$"}

titles = {"tauPt" : "Tau transverse momentum",
            "MET" : "Missing transverse energy",
            "bjetPt" : "b-jet transverse momentum",
            "ldgTrkPtFrac" : "Leading charged track momentum fraction",
            "deltaPhiTauMet" : "Angle between tau and MET",
            "deltaPhiTauBjet" : "Angle between tau and b-jet",
            "deltaPhiBjetMet" : "Angle between b-jet and MET",
            "TransverseMass" : "Invariant Transverse Mass"}

def createDistributionComparison(signal, background, columns):
    subplotColumns = 2
    subplotRows = ceil(1.0*len(columns)/subplotColumns)
    print("Subplot columns: %d, subplot rows: %d" % (subplotColumns, subplotRows))
    fig, axes = plt.subplots(subplotRows, subplotColumns, figsize=(15,20))
    ind = 0
    for column in columns:
        thisYIndex = ind % subplotColumns
        thisXIndex = floor(ind/subplotColumns)
        print("X: %d, Y: %d" %(thisXIndex, thisYIndex))
        if subplotRows > 1:
            thisAxes = axes[thisXIndex, thisYIndex]
        else:
            thisAxes = axes[thisXIndex]

        thisAxes.hist(signal[column], bins=xBinning[column], color='blue',
                                          alpha=0.7, edgecolor='black', linewidth=1.0,
                                          label="Signal", density=True)

        thisAxes.hist(background[column], bins=xBinning[column], color='green',
                                          alpha=0.7, edgecolor='black', linewidth=1.0,
                                          label="Background", density=True)
        thisAxes.set_xlabel(xLabel[column])
        thisAxes.set_title(titles[column])
        thisAxes.legend()
        ind += 1

    plt.savefig("plots/DistributionComparison.pdf")
    plt.clf()


#Assumes: Even bin widths
from scipy.stats import binned_statistic
def classifierVsX(classifier, inputData, targetData, variableName, variableData, postfix):
    binning = xBinning[variableName]
    width = binning[1]-binning[0]

    bkg = inputData[(targetData['target']==0)]
    sig = inputData[(targetData['target']==1)]
    samples = [sig, bkg]
    labels = ["signal", "background"]
    ind = 0
    variableData = [variableData[(targetData['target'] == 1)], variableData[(targetData['target'] == 0)]]
    for data in samples:
        variable = variableData[ind]
        predictions = classifier.predict(data.to_numpy()).reshape(variable.shape)
        binContent, _ = np.histogram(variable, bins=xBinning[variableName])
        weightedBins, _ = np.histogram(variable, bins=xBinning[variableName], weights=predictions)
        stdDev, _, _ = binned_statistic(variable, values=predictions, statistic='std', bins=xBinning[variableName])

        weightedBins = weightedBins[(binContent!=0)]/binContent[(binContent!=0)]
        cleanedBinning = binning[:-1][(binContent!=0)]
        stdDev[(binContent==1)] = 1.0
        stdDev = stdDev[(binContent!=0)]

        plt.errorbar(cleanedBinning+width/2, weightedBins, yerr=stdDev, fmt='.', label=labels[ind])
        ind += 1

    plt.grid(zorder=0)
    plt.xlabel(xLabel[variableName])
    plt.ylabel("Mean MVA output and std. per bin")
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 200.0)
    plt.title("Classifier output and std w.r.t. " + xLabel[variableName])
    plt.legend()

    plt.savefig("plots/ClassifierVs" + variableName + postfix + ".pdf")
    plt.clf()


