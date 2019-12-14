from utilities import createDirectories, readDatasetsToDataframes
from neuralNetworks import createChainedModel_v3, createClassifier, createAdversary, setTrainable, JSDMetric, GradientTapeCallBack, createChainedModel, StandardScalerLayer, swish
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import compute_sample_weight
from hyperOptimization import COLUMNS, PTMIN, PTMAX, PTBINS, BATCHSIZE, TESTSET_SIZE, PRETRAINEPOCHS

from sklearn.model_selection import train_test_split
from plotting import classifierVsX

import numpy as np
import tensorflow as tf

import glob

import sys
import os
from datetime import datetime

from root_pandas import read_root

tf.random.set_seed(13)
#tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly())

def main():
    columns = COLUMNS

    baseUrl = "/Users/hajohajo/Documents/repos/TrainingFiles/"
    # signalFilePaths = glob.glob(baseUrl+"ChargedHiggs*.root")
    # backgroundFilePaths = list(set(glob.glob(baseUrl+"*.root")).difference(set(signalFilePaths)))


    datasets=readDatasetsToDataframes(baseUrl)
    print(datasets)
    print(datasets[0].head())

    sys.exit(1)

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
    sampleWeights_adversary = np.ones(trainDataFrame.shape[0])
    binning = np.linspace(PTMIN, PTMAX, PTBINS+1)
    digitized = np.digitize(np.clip(trainDataFrame['TransverseMass'].values, PTMIN, PTMAX-1.0), bins=binning, right=False).astype(np.float32)
    sampleWeights_classifier[trainDataFrame.target == 0] = compute_sample_weight('balanced', digitized[trainDataFrame.target == 0])
    sampleWeights_adversary[trainDataFrame.target == 1] = 0

    # trainDataset = tf.data.Dataset.from_tensor_slices((trainDataFrame[COLUMNS].values, trainDataFrame['target'].values, sampleWeights_classifier))
    trainDataset = tf.data.Dataset.from_tensor_slices((trainDataFrame[COLUMNS].values, trainDataFrame['target'].values))
    validationDataset = trainDataset.take(TESTSET_SIZE)
    trainDataset = trainDataset.skip(TESTSET_SIZE)

    trainDataset = trainDataset.batch(BATCHSIZE, drop_remainder=True)
    validationDataset = validationDataset.batch(BATCHSIZE, drop_remainder=True)

    testDataset = tf.data.Dataset.from_tensor_slices((testDataFrame[COLUMNS].values, testDataFrame['target'].values))
    testDataset = testDataset.batch(BATCHSIZE, drop_remainder=True)

    classifierModelPath = "./models/classifierBeforeTraining.h5"
    if(os.path.exists(classifierModelPath)):
        classifier = tf.keras.models.load_model(classifierModelPath, custom_objects={"StandardScalerLayer" : StandardScalerLayer,
                                                                                        "swish" : swish})
    else:
        classifier = createClassifier(means, scale);
        classifier.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                            loss="binary_crossentropy")

        classifier.save(classifierModelPath)

    adversary = createAdversary()
    adversary.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                      loss=lambda y, model: -model.log_prob(y+1e-8))

    chainedModel = createChainedModel(classifier, adversary)
    chainedModel.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                         loss=["binary_crossentropy", lambda y, model:-model.log_prob(y+1e-8)],
                         loss_weights=[1.0, 10.0])

    print(classifier.summary())
    print(adversary.summary())
    print(chainedModel.summary())


    # logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboardCallback = tf.keras.callbacks.TensorBoard(histogram_freq=1,
                                                         update_freq='epoch',
                                                         profile_batch=0)
    gradientTapeCallback = GradientTapeCallBack(trainDataFrame[COLUMNS].to_numpy())


    sampleWeights_classifier=np.array(sampleWeights_classifier)
    sampleWeights_adversary=np.array(sampleWeights_adversary)

    classifier.fit(trainDataset,
                   epochs=PRETRAINEPOCHS,
                   validation_data=validationDataset)
    classifierVsX(classifier, testDataFrame[COLUMNS], testDataFrame, "TransverseMass", testDataFrame["TransverseMass"], "Before")


    advInput = classifier.predict(trainDataFrame[COLUMNS].to_numpy())

    adversary.fit(advInput, digitized,
                  epochs=PRETRAINEPOCHS,
                  sample_weight=sampleWeights_adversary,
                  batch_size=BATCHSIZE) #,

    chainedModel.fit(trainDataFrame[COLUMNS].to_numpy(), [trainDataFrame['target'].to_numpy(), digitized],
                     epochs=100,
                     batch_size=BATCHSIZE,
                     callbacks=[tensorboardCallback],
                     sample_weight={"classifierDense_output": sampleWeights_classifier, "Adversary": sampleWeights_adversary})
                     # callbacks=[gradientTapeCallback, tensorboardCallback])

    classifier.save("models/classifier.h5")

    print(testDataFrame.columns)
    classifierVsX(classifier, testDataFrame[COLUMNS], testDataFrame, "TransverseMass", testDataFrame["TransverseMass"], "After")

    sys.exit(1)

    jsdMetric = JSDMetric(classifierCut=0.5,
                          validation_data=(allData.sample(frac=0.1), train_target['target'].sample(frac=0.1)))


if __name__ == "__main__":
    main()