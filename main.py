from utilities import createDirectories
from neuralNetworks import createChainedModel_v3, createClassifier, createAdversary, setTrainable, JSDMetric, GradientTapeCallBack
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import compute_sample_weight
from hyperOptimization import COLUMNS, PTMIN, PTMAX, PTBINS, BATCHSIZE, TESTSET_SIZE

from sklearn.model_selection import train_test_split
from plotting import classifierVsX

import numpy as np
import tensorflow as tf

import glob

import sys
from datetime import datetime

from root_pandas import read_root

tf.random.set_seed(13)
#tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly())

def main():
    columns = COLUMNS

    baseUrl = "/Users/hajohajo/Documents/repos/TrainingFiles/"
    signalFilePaths = glob.glob(baseUrl+"ChargedHiggs*.root")
    backgroundFilePaths = list(set(glob.glob(baseUrl+"*.root")).difference(set(signalFilePaths)))

    signal = read_root(signalFilePaths, columns=columns)
    background = read_root(backgroundFilePaths, columns=columns)
    signal['target'] = 1
    background['target'] = 0

    events = min(signal.shape[0], background.shape[0])

    signal = signal.sample(n=events)
    background = background.sample(n=events)

    createDirectories()

    allData = signal.append(background, ignore_index=True)
    allData = allData[columns+["target"]]
    allData = allData.sample(frac=1.0).reset_index(drop=True)
    allData["logPt"] = np.log(allData["tauPt"].copy().values)
#    allData["unscaledTransverseMass"] = allData["TransverseMass"].copy().values

    trainDataFrame, testDataFrame = train_test_split(allData, test_size=0.1)

    scaler = StandardScaler().fit(trainDataFrame[COLUMNS])
    scale, means, vars = scaler.scale_, scaler.mean_, scaler.var_

    sampleWeights_classifier = np.ones(trainDataFrame.shape[0])
    binning = np.linspace(PTMIN, PTMAX, PTBINS+1)
    digitized = np.digitize(np.clip(trainDataFrame['TransverseMass'].values, PTMIN, PTMAX-1.0), bins=binning, right=False)
    sampleWeights_classifier[trainDataFrame.target == 0] = compute_sample_weight('balanced', digitized[trainDataFrame.target == 0])

    trainDataset = tf.data.Dataset.from_tensor_slices((trainDataFrame[COLUMNS].values, trainDataFrame['target'].values, sampleWeights_classifier))
    validationDataset = trainDataset.take(TESTSET_SIZE)
    trainDataset = trainDataset.skip(TESTSET_SIZE)

    trainDataset = trainDataset.batch(BATCHSIZE, drop_remainder=True)
    validationDataset = validationDataset.batch(BATCHSIZE, drop_remainder=True)

    testDataset = tf.data.Dataset.from_tensor_slices((testDataFrame[COLUMNS].values, testDataFrame['target'].values))
    testDataset = testDataset.batch(BATCHSIZE, drop_remainder=True)

    classifier = createClassifier(means, scale);
    classifier.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                        loss="binary_crossentropy")


    # logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboardCallback = tf.keras.callbacks.TensorBoard(histogram_freq=1,
                                                         update_freq='epoch',
                                                         profile_batch=0)
    gradientTapeCallback = GradientTapeCallBack(trainDataFrame)

    classifier.fit(trainDataset,
                   epochs=15,
                   validation_data=validationDataset,
                   callbacks=[gradientTapeCallback, tensorboardCallback])

    sys.exit(1)

    weights = np.ones(allData.shape[0])
    binning = np.linspace(PTMIN, PTMAX, PTBINS+1)
    digitized = np.digitize(np.clip(allData['TransverseMass'].values, PTMIN, PTMAX-1.0), bins=binning, right=False) - 1

    allData_target = allData['target'].values
    allData_adversarialTarget = allData['unscaledTransverseMass'].values

    weights[allData.target==0] = compute_sample_weight('balanced', digitized[allData.target==0])
    allData["weights"] = weights

    scaler2 = MinMaxScaler()

    allData["adversarialTarget"] = allData_adversarialTarget
    allData["adversarialTarget"] = scaler2.fit_transform(allData["adversarialTarget"].to_numpy().reshape(-1, 1))
    train_input, test_input, train_target, test_target = train_test_split(allData[columns+["logPt", "weights", "unscaledTransverseMass"]],
                                                                          allData[["target", "adversarialTarget"]],
                                                                          test_size=TESTSET_SIZE)

    allData = pd.DataFrame(train_input, columns=COLUMNS+["logPt", "weights", "unscaledTransverseMass"])
    allTarget = pd.DataFrame(train_target, columns=["target", "adversarialTarget"])
    testDataFrame = pd.DataFrame(test_input, columns=COLUMNS+["logPt", "weights"])

    scaler1 = MinMaxScaler()
    scaler3 = MinMaxScaler()
    allData["logPt"] = scaler3.fit_transform(allData["logPt"].to_numpy().reshape(-1, 1))
    testDataFrame["logPt"] = scaler3.transform(testDataFrame["logPt"].to_numpy().reshape(-1, 1))

    classifierWeights = train_input["weights"].to_numpy()
    adversaryWeights = np.array(np.multiply(train_input["weights"], np.logical_not(train_target["target"])))



    jsdMetric = JSDMetric(classifierCut=0.5,
                          validation_data=(allData.sample(frac=0.1), train_target['target'].sample(frac=0.1)))

    classifier = createClassifier()
    classifier.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                        loss="binary_crossentropy")

    adversary = createAdversary()
    chained3 = createChainedModel_v3(classifier, adversary, 10.0)

    chained3.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                     loss=["binary_crossentropy", lambda y, model:-model.log_prob(y+1e-8)],
                     weights=[1.0, 1.0])

    setTrainable(classifier, False)
    chainedFreeze = createChainedModel_v3(classifier, adversary, 10.0)
    chainedFreeze.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                     loss=["binary_crossentropy", lambda y, model:-model.log_prob(y+1e-8)],
                     weights=[1.0, 1.0])





    variableData = allData["unscaledTransverseMass"].copy()

    classifier.fit(allData[COLUMNS].to_numpy(), allTarget['target'].to_numpy(),
                    epochs=100,
                    batch_size=BATCHSIZE,
                    sample_weight=classifierWeights,
                    validation_split=0.05)
    classifierVsX(classifier, allData[COLUMNS], allTarget, "TransverseMass", variableData, "Before")

    chainedFreeze.fit([allData[COLUMNS].to_numpy(), allData["logPt"].to_numpy()], [allTarget['target'].to_numpy(), allTarget['adversarialTarget'].to_numpy()],
                 epochs=100,
                 batch_size=BATCHSIZE,
                 sample_weight={"out_classifier": classifierWeights, "Adversary": adversaryWeights},
                 # sample_weight=[classifierWeights, adversaryWeights],
                 callbacks=[jsdMetric])

    chained3.fit([allData[COLUMNS].to_numpy(), allData["logPt"].to_numpy()], [allTarget['target'].to_numpy(), allTarget['adversarialTarget'].to_numpy()],
                 epochs=300,
                 batch_size=BATCHSIZE,
                 # sample_weight=[classifierWeights, adversaryWeights],
                 sample_weight={"out_classifier":classifierWeights, "Adversary":adversaryWeights},
                 callbacks=[jsdMetric])

    classifierVsX(classifier, train_input[COLUMNS], train_target, "TransverseMass", variableData, "After")



if __name__ == "__main__":
    main()