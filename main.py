import matplotlib
matplotlib.use('Agg')
from utilities import createDirectories, readDatasetsToDataframes, quantileClipper, getClassifierSampleWeights, getAdversarySampleWeights
from neuralNetworks import createChainedModel, StandardScalerLayer, swish, createMultiClassifier, createMultiAdversary
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hyperOptimization import COLUMNS_, BATCHSIZE, PRETRAINEPOCHS, TRAINEPOCHS, PTMIN, PTMAX, PTBINS

from sklearn.model_selection import train_test_split
from plotting import classifierVsX, multiClassClassifierVsX, jsdScoresMulti

import numpy as np
import tensorflow as tf

import sys
from datetime import datetime


tf.random.set_seed(13)

print(tf.executing_eagerly())

#USEPRETRAINED = True
USEPRETRAINED = False
pretrainedClassifierModelPath = "./models/classifierSaved.h5"
pretrainedAdversaryModelPath = "./models/adversarySaved.h5"

def main():
    baseUrl = "/work/hajohajo/TrainingFiles/"
    #baseUrl = "/Users/hajohajo/Documents/repos/TrainingFiles/"

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
    adversaryTargets = MinMaxScaler().fit_transform(trainingInputPreprocessed.loc[:,"TransverseMass"].to_numpy().reshape(-1, 1))

    auxiliaryTrainingInput = MinMaxScaler().fit_transform(np.log(trainingInputPreprocessed.loc[:, ["tauPt", "MET"]]))
    auxiliaryTestInputs = np.log(testInputPreprocessed.loc[:, ["tauPt", "MET"]])

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
                          loss=lambda y, model: -model.log_prob(y))

        chainedModel = createChainedModel(classifier, adversary)
        chainedModel.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3, amsgrad=True),
                             loss=["categorical_crossentropy", lambda y, model:-model.log_prob(y)],
                             loss_weights=[1.0, 10.0])

        classifier.fit(trainingInputPreprocessed.loc[:, COLUMNS_].to_numpy(), classifierTargets,   #trainDataset,
                       epochs=PRETRAINEPOCHS,
                       batch_size=BATCHSIZE,
                       sample_weight=sampleWeights_classifier,
                       validation_split=0.1)
        classifier.save("models/classifier"+datetime.now().strftime("%Y%m%d-%H%M%S")+".h5")

        jsdScoresMulti(classifier, testInputPreprocessed.loc[:, COLUMNS_],testDF, testDF.loc[:, "TransverseMass"], "Before")
        multiClassClassifierVsX(classifier, testInputPreprocessed.loc[:, COLUMNS_], testDF, "TransverseMass", testDF.loc[:, "TransverseMass"], "Before")
        classifierVsX(classifier, testInputPreprocessed.loc[:, COLUMNS_], testDF, "TransverseMass", testDF.loc[:, "TransverseMass"], "BeforeAllBkgs")


        adversaryInput = classifier.predict(trainingInputPreprocessed.loc[:, COLUMNS_].to_numpy())
#        adversary.fit(adversaryInput, adversaryTargets,
        adversary.fit([adversaryInput, auxiliaryTrainingInput], adversaryTargets,
                      epochs=PRETRAINEPOCHS,
                      sample_weight=sampleWeights_adversary,
                      batch_size=BATCHSIZE,
                      validation_split=0.1)
        adversary.save("models/adversary"+datetime.now().strftime("%Y%m%d-%H%M%S")+".h5")

    else:
        classifier = tf.keras.models.load_model(pretrainedClassifierModelPath,
                                                custom_objects={"StandardScalerLayer": StandardScalerLayer, "swish": swish})

        jsdScoresMulti(classifier, testInputPreprocessed.loc[:, COLUMNS_],testDF, testDF.loc[:, "TransverseMass"], "Before")
        multiClassClassifierVsX(classifier, testInputPreprocessed.loc[:, COLUMNS_], testDF, "TransverseMass", testDF.loc[:, "TransverseMass"], "Before")
        classifierVsX(classifier, testInputPreprocessed.loc[:, COLUMNS_], testDF, "TransverseMass", testDF.loc[:, "TransverseMass"], "BeforeAllBkgs")


        adversary = tf.keras.models.load_model(pretrainedAdversaryModelPath,
                                               custom_objects={'<lambda>': lambda y, model:-model.log_prob(y)})

    initial_learning_rate = 1e-5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

#    chainedModel = createChainedModel(classifier, adversary)
#    chainedModel.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True),
#                         loss=["categorical_crossentropy", lambda y, model:-model.log_prob(y)],
#                         loss_weights=[1.0, 0.0]) #5.0])

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

    chainedModel.fit([trainingInputPreprocessed.loc[:, COLUMNS_].to_numpy(),auxiliaryTrainingInput] , [classifierTargets, adversaryTargets],
#    chainedModel.fit([trainingInputPreprocessed.loc[:, COLUMNS_].to_numpy()] , [classifierTargets, adversaryTargets],
                     epochs=TRAINEPOCHS,
                     batch_size=BATCHSIZE,
                     sample_weight={"classifierDense_output": sampleWeights_classifier, "Adversary": sampleWeights_adversary})
                     # callbacks=[gradientTapeCallback, tensorboardCallback])


    classifierVsX(classifier, testInputPreprocessed.loc[:, COLUMNS_], testDF, "TransverseMass", testDF.loc[:, "TransverseMass"], "AfterAllBkgsTest"+datetime.now().strftime("%Y%m%d-%H%M%S"))
    jsdScoresMulti(classifier, testInputPreprocessed.loc[:, COLUMNS_],testDF, testDF.loc[:, "TransverseMass"], "AfterTest"+datetime.now().strftime("%Y%m%d-%H%M%S"))
    multiClassClassifierVsX(classifier, testInputPreprocessed.loc[:, COLUMNS_], testDF, "TransverseMass", testDF.loc[:, "TransverseMass"], "AfterTest"+datetime.now().strftime("%Y%m%d-%H%M%S"))

    classifier.save("models/classifier.h5")

if __name__ == "__main__":
    main()
