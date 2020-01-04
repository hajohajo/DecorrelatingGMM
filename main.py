import matplotlib
matplotlib.use('Agg')
from utilities import createDirectories, readDatasetsToDataframes, quantileClipper, getClassifierSampleWeights, getAdversarySampleWeights
from neuralNetworks import createChainedModel, StandardScalerLayer, swish, createMultiClassifier, createMultiAdversary
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hyperOptimization import COLUMNS_, BATCHSIZE, PRETRAINEPOCHS, PTMIN, PTMAX, PTBINS

from sklearn.model_selection import train_test_split
from plotting import classifierVsX, multiClassClassifierVsX, jsdScoresMulti

import numpy as np
import tensorflow as tf

import sys
from datetime import datetime


tf.random.set_seed(23)

print(tf.executing_eagerly())

TRAINEPOCHS=1
#USEPRETRAINED = True
USEPRETRAINED = False
pretrainedClassifierModelPath = "./models/classifierSaved.h5"
pretrainedAdversaryModelPath = "./models/adversarySaved.h5"

def main():
    # baseUrl = "/work/hajohajo/TrainingFiles/"
    baseUrl = "/Users/hajohajo/Documents/repos/TrainingFiles/"

    dataframes = readDatasetsToDataframes(baseUrl)
    createDirectories()

    dataframes = dataframes.sample(frac=1.0).reset_index(drop=True)

    trainingDF, testDF = train_test_split(dataframes, test_size=0.2)

    clipper = quantileClipper(lowerQuantile=0.1, upperQuantile=0.9)
    clipper.fit(trainingDF.loc[:, COLUMNS_])

    trainingInputPreprocessed = trainingDF.copy()
    testInputPreprocessed = testDF.copy()
    trainingInputPreprocessed.loc[:, COLUMNS_] = clipper.clip(trainingDF.loc[:, COLUMNS_])
    testInputPreprocessed.loc[:, COLUMNS_] = clipper.clip(testDF.loc[:, COLUMNS_])
    classifierTargets = tf.keras.utils.to_categorical(trainingInputPreprocessed.loc[:, "eventType"].values)
    adversaryTargets = tf.keras.utils.to_categorical(np.digitize(np.clip(trainingInputPreprocessed.loc[:, "TransverseMass"], PTMIN, PTMAX-1.0),
                                                                 bins=np.linspace(PTMIN, PTMAX, PTBINS+1)))

    scaler = StandardScaler().fit(trainingInputPreprocessed[COLUMNS_])
    scale, means, vars = scaler.scale_, scaler.mean_, scaler.var_

    sampleWeights_classifier = getClassifierSampleWeights(trainingInputPreprocessed)
    sampleWeights_adversary = getAdversarySampleWeights(trainingInputPreprocessed)

    if not USEPRETRAINED:
        classifier = createMultiClassifier(means, scale)
        classifier.compile(
                            optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                            loss="categorical_crossentropy")

        adversary = createMultiAdversary()
        adversary.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                          loss=lambda y, model: -model.log_prob(y+1e-6))

        classifier.fit(trainingInputPreprocessed.loc[:, COLUMNS_].to_numpy(), classifierTargets,   #trainDataset,
                       epochs=PRETRAINEPOCHS,
                       batch_size=BATCHSIZE,
                       validation_split=0.1)
        classifier.save("models/classifier"+datetime.now().strftime("%Y%m%d-%H%M%S")+".h5")

        adversaryInput = classifier.predict(trainingInputPreprocessed.loc[:, COLUMNS_].to_numpy())
        adversary.fit(adversaryInput, adversaryTargets,
                      epochs=PRETRAINEPOCHS,
                      sample_weight=sampleWeights_adversary,
                      batch_size=BATCHSIZE,
                      validation_split=0.1)
        adversary.save("models/adversary"+datetime.now().strftime("%Y%m%d-%H%M%S")+".h5")

    else:
        classifier = tf.keras.models.load_model(pretrainedClassifierModelPath,
                                                custom_objects={"StandardScalerLayer": StandardScalerLayer, "swish": swish})
        adversary = tf.keras.models.load_model(pretrainedAdversaryModelPath,
                                               custom_objects={'<lambda>': lambda y, model:-model.log_prob(y+1e-6)})

    jsdScoresMulti(classifier, testInputPreprocessed.loc[:, COLUMNS_],testDF, testDF.loc[:, "TransverseMass"], "Before")
    multiClassClassifierVsX(classifier, testInputPreprocessed.loc[:, COLUMNS_], testDF, "TransverseMass", testDF.loc[:, "TransverseMass"], "Before")
    classifierVsX(classifier, testInputPreprocessed.loc[:, COLUMNS_], testDF, "TransverseMass", testDF.loc[:, "TransverseMass"], "BeforeAllBkgs")

    chainedModel = createChainedModel(classifier, adversary)
    chainedModel.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),
                         loss=["categorical_crossentropy", lambda y, model:-model.log_prob(y+1e-6)],
                         loss_weights=[1.0, 10.0])

    # print(classifier.summary())
    # print(adversary.summary())
    # print(chainedModel.summary())


    # logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #
    # tensorboardCallback = tf.keras.callbacks.TensorBoard(histogram_freq=1,
    #                                                      update_freq='epoch',
    #                                                      profile_batch=0)
    # gradientTapeCallback = GradientTapeCallBack(trainDataFrame[COLUMNS].to_numpy())
    #

    chainedModel.fit(clipper.clip(trainingInputPreprocessed.loc[:, COLUMNS_]).to_numpy(), [classifierTargets, adversaryTargets],
                     epochs=TRAINEPOCHS,
                     batch_size=BATCHSIZE,
                     sample_weight={"classifierDense_output": sampleWeights_classifier, "Adversary": sampleWeights_adversary})
                     # callbacks=[gradientTapeCallback, tensorboardCallback])

    classifier.save("models/classifier.h5")

    jsdScoresMulti(classifier, testInputPreprocessed.loc[:, COLUMNS_],testDF, testDF.loc[:, "TransverseMass"], "After")
    multiClassClassifierVsX(classifier, testInputPreprocessed.loc[:, COLUMNS_], testDF, "TransverseMass", testDF.loc[:, "TransverseMass"], "After")
    classifierVsX(classifier, testInputPreprocessed.loc[:, COLUMNS_], testDF, "TransverseMass", testDF.loc[:, "TransverseMass"], "AfterAllBkgs")

if __name__ == "__main__":
    main()
