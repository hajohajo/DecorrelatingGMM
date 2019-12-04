from utilities import createDirectories
from neuralNetworks import createChainedModel_v3, createClassifier, createAdversary, setTrainable
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_sample_weight
from hyperOptimization import COLUMNS, PTMIN, PTMAX, PTBINS, BATCHSIZE

from sklearn.model_selection import train_test_split
from plotting import classifierVsX

import numpy as np
import tensorflow as tf
import glob

from root_pandas import read_root

tf.random.set_seed(13)
#tf.compat.v1.disable_eager_execution()


def main():
    TESTSET_SIZE = 10000
    columns = COLUMNS

    baseUrl = "/Users/hajohajo/Documents/repos/TrainingFiles/"
    signalFilePaths = glob.glob(baseUrl+"ChargedHiggs*.root")
    backgroundFilePaths = list(set(glob.glob(baseUrl+"*.root")).difference(set(signalFilePaths)))

    signal = read_root(signalFilePaths, columns=columns)
    background = read_root(backgroundFilePaths, columns=columns)
    signal['target'] = 1
    background['target'] = 0

    print(signal.shape)
    print(background.shape)

    events = min(signal.shape[0], background.shape[0])

    signal = signal.sample(n=events)
    background = background.sample(n=events)


    createDirectories()

    allData = signal.append(background, ignore_index=True)
    allData = allData[columns+["target"]]
    allData = allData.sample(frac=1.0).reset_index(drop=True)
    allData["logPt"] = np.log(allData["tauPt"].copy().values)
    allData["unscaledTransverseMass"] = allData["TransverseMass"].copy().values

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
    testData = pd.DataFrame(test_input, columns=COLUMNS+["logPt", "weights"])

    scaler1 = MinMaxScaler()
    scaler3 = MinMaxScaler()
    allData["logPt"] = scaler3.fit_transform(allData["logPt"].to_numpy().reshape(-1, 1))
    testData["logPt"] = scaler3.transform(testData["logPt"].to_numpy().reshape(-1, 1))

    classifierWeights = train_input["weights"].to_numpy()
    adversaryWeights = np.array(np.multiply(train_input["weights"], np.logical_not(train_target["target"])))

    class JSDMetric(tf.keras.callbacks.Callback):
        def __init__(self, classifierCut, validation_data):
            super().__init__()
            self.JSDScores = []
            self.validation_data = validation_data
            self.classifierCut = classifierCut
            self.epoch = 0

        def getJSD(self):
            input, target = self.validation_data
            masses = input.unscaledTransverseMass.to_numpy().reshape(-1,1)
            auxiliary = input["logPt"].to_numpy()
            input = input[COLUMNS].to_numpy()
            target = target.to_numpy()
            input = input[(target == 0)]
            auxiliary = auxiliary[(target == 0)]
            masses = masses[(target == 0)]
            predictions = self.model.predict([input, auxiliary])[0]
            passArray = np.array(predictions > self.classifierCut, dtype=bool)
            failArray = np.array(predictions <= self.classifierCut, dtype=bool)
            passed = masses[passArray]
            failed = masses[failArray]

            binContentPassed, _ = np.histogram(passed, bins=PTBINS, density=True)
            binContentFailed, _ = np.histogram(failed, bins=PTBINS, density=True)

            k = tf.keras.losses.KLDivergence()
            m = (binContentPassed + binContentFailed)/2.0

            JSDScore = (k(binContentPassed, m) + k(binContentFailed, m))/2.0

            return JSDScore.numpy()

        def on_epoch_end(self, epoch, logs={}):
            if(self.epoch%10==0):
                score = self.getJSD()
                print("\t - JSD: %f" % (score))
                self.JSDScores.append(score)

            self.epoch += 1


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