import numpy as np
from math import ceil, floor
import seaborn as sns
import matplotlib.pyplot as plt
from hyperOptimization import PTBINS, PTMIN, PTMAX

sns.set()
sns.set_style("whitegrid")

xBinning = {"tauPt" : np.linspace(0.0, 500.0, 100),
            "MET" : np.linspace(0.0, 500.0, 100),
            "bjetPt" : np.linspace(0.0, 500.0, 100),
            "ldgTrkPtFrac" : np.linspace(0.0, 1.0, 20),
            "deltaPhiTauMet" : np.linspace(0.0, np.pi, 30),
            "deltaPhiTauBjet" : np.linspace(0.0, np.pi, 30),
            "deltaPhiBjetMet" : np.linspace(0.0, np.pi, 30),
#            "TransverseMass" : np.linspace(0.0, 1000.0, 20)}
            "TransverseMass": np.linspace(0.0, 300.0, 60)}

xLabel = {"tauPt" : r"Tau p$_T$",
            "MET" : r"E$_{T, miss}$",
            "bjetPt" : r"B-jet p$_T$",
            "ldgTrkPtFrac" : r"Leading charged track p$_T$",
            "deltaPhiTauMet" : r"$\Delta\Phi_{\tau, MET}$",
            "deltaPhiTauBjet" : r"$\Delta\Phi_{\tau, MET}$",
            "deltaPhiBjetMet" : r"$\Delta\Phi_{b jet, MET}$",
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
    fig, axes = plt.subplots(subplotRows, subplotColumns, figsize=(15,30))
    ind = 0
    for column in columns:
        thisYIndex = ind % subplotColumns
        thisXIndex = floor(ind/subplotColumns)
        print("X: %d, Y: %d" %(thisXIndex, thisYIndex))
        if subplotRows > 1:
            thisAxes = axes[thisXIndex, thisYIndex]
        else:
            thisAxes = axes[thisXIndex]

        thisAxes.hist(signal[column], bins=xBinning[column], color=sns.xkcd_rgb['teal'],
                                          alpha=0.7, edgecolor='black', linewidth=1.0,
                                          label="Signal", density=True)

        thisAxes.hist(background[column], bins=xBinning[column], color=sns.xkcd_rgb['crimson'],
                                          alpha=0.7, edgecolor='black', linewidth=1.0,
                                          label="Background", density=True)
        thisAxes.set_xlabel(xLabel[column])
        thisAxes.set_title(titles[column])
        thisAxes.legend()
        thisAxes.set_xlim(xBinning[column][0], xBinning[column][-1])
        thisAxes.set_ylabel(ylabel="Fraction of events")
        ind += 1

    plt.savefig("plots/DistributionComparison.pdf")
    plt.clf()


#Assumes: Even bin widths
from scipy.stats import binned_statistic
def classifierVsX(classifier, inputData, targetData, variableName, variableData, postfix):
    binning = xBinning[variableName]
    width = binning[1]-binning[0]

    # targetData['target'] = np.argmax(targetData['eventType'].values)==0
    targetData['target'] = (targetData['eventType']==0)

    bkg = inputData[(targetData['target']==0)]
    sig = inputData[(targetData['target']==1)]

    samples = [sig, bkg]
    labels = ["signal", "background"]
    colors = [sns.xkcd_rgb["teal"], sns.xkcd_rgb["crimson"]]
    ind = 0
    variableData = [variableData[(targetData['target'] == 1)], variableData[(targetData['target'] == 0)]]
    for data in samples:
        variable = variableData[ind]
        # predictions = classifier.predict(data.to_numpy())[0].reshape(variable.shape)
        predictions = classifier.predict(data.to_numpy())
        predictions = predictions[:, 0]
        binContent, _ = np.histogram(variable, bins=xBinning[variableName])
        weightedBins, _ = np.histogram(variable, bins=xBinning[variableName], weights=predictions)
        stdDev, _, _ = binned_statistic(variable, values=predictions, statistic='std', bins=xBinning[variableName])

        weightedBins = weightedBins[(binContent!=0)]/binContent[(binContent!=0)]
        cleanedBinning = binning[:-1][(binContent!=0)]
        stdDev[(binContent==1)] = 1.0
        stdDev = stdDev[(binContent!=0)]

        plt.scatter(cleanedBinning+width/2, weightedBins, marker='.', label=labels[ind], color=colors[ind], alpha=0.7, linewidths=1.0, edgecolors='k')
        extended = np.append(np.insert(weightedBins, 0, weightedBins[0]-(weightedBins[1]-weightedBins[0])/2.0), weightedBins[-1]+(weightedBins[-1]-weightedBins[-2])/2.0)
        up = extended+np.append(np.insert(stdDev, 0, stdDev[0]),stdDev[-1])
        down = extended-np.append(np.insert(stdDev, 0, stdDev[0]),stdDev[-1])
        extendedBinning = np.append(np.insert(cleanedBinning+width/2, 0, cleanedBinning[0]), cleanedBinning[-1]+width)
        plt.plot(extendedBinning, up, color=colors[ind], alpha=0.9)
        plt.plot(extendedBinning, down, color=colors[ind], alpha=0.9)
        plt.fill_between(extendedBinning, up, down, label="$\pm 1\sigma$", alpha=0.6, color=colors[ind])
        ind += 1

    plt.grid(zorder=0)
    plt.xlabel(xLabel[variableName])
    plt.ylabel("Mean MVA output per bin")
    plt.ylim(0.0, 1.0)
    plt.xlim(binning[0],binning[-1]+width)
    plt.title("Classifier output and std w.r.t. " + xLabel[variableName])
    plt.legend()

    plt.savefig("plots/ClassifierVs" + variableName + postfix + ".pdf")
    plt.clf()

from utilities import invertedEventTypeDict
def multiClassClassifierVsX(classifier, inputData, targetData, variableName, variableData, postfix):
    binning = xBinning[variableName]
    width = binning[1]-binning[0]

    targetData.loc[:, 'target'] = targetData.loc[:, 'eventType']

    sig = [inputData.loc[targetData['eventType'] == 0]]
    bkgs = []
    for i in range(1, len(invertedEventTypeDict)):
        bkgs.append(inputData.loc[targetData['eventType'] == i])

    samples = sig+bkgs
    labels = [invertedEventTypeDict[x] for x in range(0, len(invertedEventTypeDict))]
    colors = [sns.xkcd_rgb["teal"], sns.xkcd_rgb["crimson"], 'k','b','r','gray','y']
    ind = 0
    variableData = [variableData.loc[(targetData['target'] == x)] for x in range(0, len(invertedEventTypeDict))]

    for data in samples:
        if(data.empty or data.shape[0]==1):
            ind += 1
            continue
        variable = variableData[ind]
        predictions = classifier.predict(data.to_numpy())
        predictions = predictions[:, 0]
        binContent, _ = np.histogram(variable, bins=xBinning[variableName])
        weightedBins, _ = np.histogram(variable, bins=xBinning[variableName], weights=predictions)
        stdDev, _, _ = binned_statistic(variable, values=predictions, statistic='std', bins=xBinning[variableName])

        weightedBins = weightedBins[(binContent!=0)]/binContent[(binContent!=0)]
        cleanedBinning = binning[:-1][(binContent!=0)]
        stdDev[(binContent==1)] = 1.0
        stdDev = stdDev[(binContent!=0)]


        # plt.errorbar(cleanedBinning+width/2, weightedBins, yerr=stdDev, marker='.', label=labels[ind], color=colors[ind], alpha=0.7)
        plt.scatter(cleanedBinning+width/2, weightedBins, marker='.', label=labels[ind], color=colors[ind], alpha=0.7, linewidths=1.0, edgecolors='k')
        if(len(weightedBins)<2):
            extended = np.append(np.insert(weightedBins, 0, weightedBins[0] - width/2), weightedBins[0] + width /2)
        else:
            extended = np.append(np.insert(weightedBins, 0, weightedBins[0]-(weightedBins[1]-weightedBins[0])/2.0), weightedBins[-1]+(weightedBins[-1]-weightedBins[-2])/2.0)
        up = extended+np.append(np.insert(stdDev, 0, stdDev[0]),stdDev[-1])
        down = extended-np.append(np.insert(stdDev, 0, stdDev[0]),stdDev[-1])
        extendedBinning = np.append(np.insert(cleanedBinning+width/2, 0, cleanedBinning[0]), cleanedBinning[-1]+width)

        plt.plot(extendedBinning, up, color=colors[ind], alpha=0.8, linewidth=0.6)
        plt.plot(extendedBinning, down, color=colors[ind], alpha=0.8, linewidth=0.6)
        # plt.fill_between(extendedBinning, up, down, label="$\pm 1\sigma$", alpha=0.6, color=colors[ind])
        # plt.fill_between(extendedBinning, up, down, alpha=0.3, color=colors[ind])
        ind += 1

    plt.grid(zorder=0)
    plt.xlabel(xLabel[variableName])
    plt.ylabel("Mean MVA output per bin")
    plt.ylim(0.0, 1.0)
    plt.xlim(binning[0],binning[-1]+width)
    plt.title("Classifier output and std w.r.t. " + xLabel[variableName])
    plt.legend()

    plt.savefig("plots/ClassifierVs" + variableName + postfix + ".pdf")
    plt.clf()
