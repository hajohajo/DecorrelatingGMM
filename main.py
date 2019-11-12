from generateDummyData import generateSamples
from utilities import createDirectories
from plotting import createDistributionComparison
from neuralNetworks import createChainedModel_v2, createChainedModel_v3, trainingLoop, train_test_Classifier, createClassifier, createAdversary, createAdversary, JensenShannonDivergence, createChainedModel
import pandas as pd
from tensorboard.plugins.hparams import api as hp
from hyperOptimization import COLUMNS, HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, METRIC_ACCURACY, HP_ACTIVATION, HP_NUM_HIDDEN_LAYERS, BATCHSIZE, PTBINS, PTMIN, PTMAX
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight, compute_sample_weight

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
import neural_structured_learning as nsl

tf.random.set_seed(13)
#tf.compat.v1.disable_eager_execution()


def setTrainable(model, isTrainable):
    model.trainable = isTrainable
    for layer in model.layers:
        layer.trainable = isTrainable

def run(run_dir, hparams, train_data, test_data):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = train_test_Classifier(hparams, train_data, test_data)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

def hyperOptimizeClassifier(train_data, test_data):
    session_num = 0
    for num_units in range(HP_NUM_UNITS.domain.min_value, HP_NUM_UNITS.domain.max_value, 8):
        print(num_units)
        for dropout_rate in np.linspace(HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value, 5):
            print(dropout_rate)
            for optimizer in HP_OPTIMIZER.domain.values:
                print(optimizer)
                for num_layers in range(HP_NUM_HIDDEN_LAYERS.domain.min_value, HP_NUM_HIDDEN_LAYERS.domain.max_value, 2):
                    print(num_layers)
                    for activation in HP_ACTIVATION.domain.values:
                        print(activation)
                        hparams = {
                            HP_NUM_UNITS: num_units,
                            HP_DROPOUT: dropout_rate,
                            HP_OPTIMIZER: optimizer,
                            HP_NUM_HIDDEN_LAYERS: num_layers,
                            HP_ACTIVATION: activation,
                        }
                        run_name = "run-%d" % session_num
                        print("--- Starting trial: %s" % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run('logs/hparam_tuning/' + run_name, hparams, train_data, test_data)
                        session_num += 1

def main():
    TESTSET_SIZE = 10000
#    columns = ["MET", "tauPt", "ldgTrkPtFrac", "deltaPhiTauMet", "deltaPhiTauBjet", "bjetPt", "deltaPhiBjetMet", "TransverseMass"]
    columns = COLUMNS

    createDirectories()
    signal = generateSamples(1000, isSignal=True)
    background = generateSamples(50000, isSignal=False)
#    createDistributionComparison(signal, background, columns=columns)
    allData = signal.append(background, ignore_index=True)
    print(allData.columns.values)
    allData=allData[columns+["target"]]
    allData = allData.sample(frac=1.0).reset_index(drop=True)
    allData["logPt"] = np.log(allData["tauPt"].copy().values)

    binning = np.linspace(PTMIN, PTMAX, PTBINS+1)
    digitized = np.digitize(np.clip(allData['TransverseMass'].values, PTMIN, PTMAX-1.0), bins=binning, right=False) - 1

    allData_target = allData['target'].values
    allData_adversarialTarget = allData['TransverseMass'].values

    weights = compute_sample_weight('balanced', digitized)
    allData["weights"] = weights

    # plt.scatter(allData_input[:10000], allData_target[:10000])
    # plt.xlim(0.0, 200.0)
    # plt.ylim(0.0, 200.0)
    # plt.savefig("plots/inputsAndTargets.pdf")
    # plt.clf()

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    scaler3 = MinMaxScaler()
    allData["logPt"] = scaler3.fit_transform(allData["logPt"].to_numpy().reshape(-1,1))
#    allData[columns] = scaler1.fit_transform(allData[columns])

#    allData_input = scaler1.fit_transform(allData_input.reshape(-1, 1))
    allData["adversarialTarget"] = scaler2.fit_transform(allData_adversarialTarget.reshape(-1, 1))

    print(allData.columns.values)
    train_input, test_input, train_target, test_target = train_test_split(allData[columns+["logPt", "weights"]],
                                                                          allData[["target", "adversarialTarget"]],
                                                                          test_size=TESTSET_SIZE)

    # datasetTrain = tf.data.Dataset.from_tensor_slices((allData_input.values, allData_target.values.reshape((-1, 1))))
    # print(allData_input.shape, allData_target.shape, weights.shape)
    # datasetTrain = tf.data.Dataset.from_tensor_slices((allData_input, allData_target, weights))
    #
    # datasetTest = datasetTrain.take(TESTSET_SIZE)
    # datasetTest_target = allData_target[:TESTSET_SIZE]
    # datasetTest_adversarialTarget = allData_adversarialTarget[TESTSET_SIZE]
    # datasetTest_logPt = allData_logPt
    # datasetTrain = datasetTrain.skip(TESTSET_SIZE)
    #
    # datasetTest = datasetTest.batch(BATCHSIZE)
    # datasetTrain = datasetTrain.batch(BATCHSIZE)

#    trainingLoop(classifier, chained3, train_input, train_target, 10)

    classifierWeights = train_input["weights"].to_numpy()
    adversaryWeights = np.array(np.multiply(train_input["weights"], np.logical_not(train_target["target"])))


    print(train_target["target"][:5])
    print(adversaryWeights[:5])

    # from tensorflow.keras.metrics import Metric
    # tf.config.experimental_run_functions_eagerly(True)
    # class JSDMetric(Metric):
    #     classifierCut = 1.0
    #     def __init(self, name="jensen_shannon_divergence", **kwargs):
    #         super(JensenShannonDivergence, self).__init__(name=name, **kwargs)
    #         self.classifierCut = 0.5
    #         self.JSD = 1.0
    #
    #     def update_state(self, y_true, y_pred, sample_weight=None):
    #         y_true = y_true.numpy()
    #         y_pred = y_pred.numpy()
    #         print(y_true)
    #         print(y_pred)
    #         backgroundPred = y_pred[(y_true==1)]
    #         passed = backgroundPred(backgroundPred>self.classifierCut)
    #         failed = backgroundPred(backgroundPred<=self.classifierCut)
    #         binContentPassed, _ = np.histogram(passed, bins=binning, density=True)
    #         binContentFailed, _ = np.histogram(failed, bins=binning, density=True)
    #
    #         k = tf.keras.losses.KLDivergence()
    #         m = (binContentPassed + binContentFailed)/2.0
    #         JSDScore = (k(binContentPassed, m) + k(binContentFailed, m))/2.0
    #
    #         self.JSD = JSDScore
    #
    #     def result(self):
    #         return self.JSD
    #
    #     def reset_states(self):
    #         self.JSD=1.0


    class JSDMetric(tf.keras.callbacks.Callback):
        def __init__(self, classifierCut, validation_data):
            super().__init__()
            self.JSDScores = []
            self.validation_data = validation_data
            self.classifierCut = classifierCut
            self.epoch = 0

        def getJSD(self):
            input, target = self.validation_data
            masses = input.TransverseMass.to_numpy().reshape(-1,1)
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
            score = self.getJSD()
            print("\t - JSD: %f" % (score))
            self.JSDScores.append(score)
            self.epoch += 1


    jsdMetric = JSDMetric(classifierCut=0.5,
                          validation_data=(train_input, train_target['target']))

    classifier = createClassifier()
    classifier.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                        loss="binary_crossentropy")

    adversary = createAdversary()
    chained3 = createChainedModel_v3(classifier, adversary, 10.0)

    chained3.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                     loss=["binary_crossentropy", lambda y, model:-model.log_prob(y+1e-6)])



    classifier.fit(train_input[COLUMNS].to_numpy(), train_target['target'].to_numpy(),
                    epochs=5,
                    batch_size=BATCHSIZE,
                    sample_weight=train_input["weights"].to_numpy())
    chained3.fit([train_input[COLUMNS].to_numpy(), train_input["logPt"].to_numpy()], [train_target['target'].to_numpy(), train_target['adversarialTarget'].to_numpy()],
                 epochs=2,
                 batch_size=BATCHSIZE,
                 sample_weight=[classifierWeights, adversaryWeights],
                 callbacks=[jsdMetric])


    # predictions = classifier.predict(test_input[COLUMNS].to_numpy())
    #
    # passed = test_input[(predictions > 0.5)].TransverseMass
    # failed = test_input[(predictions <= 0.5)].TransverseMass
    #
    # binContentPassed, _ = np.histogram(passed, bins=binning, density=True)
    # binContentFailed, _ = np.histogram(failed, bins=binning, density=True)
    #
    # k = tf.keras.losses.KLDivergence()
    # m = (binContentPassed + binContentFailed)/2.0
    # JSDScore = (k(binContentPassed, m) + k(binContentFailed, m))/2.0

    # chainedModel = createChainedModel(classifier)
    # setTrainable(classifier, False)
    # chainedModel.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5),
    #                    loss=lambda y, model:-model.log_prob(y + 1e-6))


#    trainingLoop(classifier, adversary, train_input, train_target, 10)

    # classifier.fit(train_input[columns].to_numpy(),
    #                train_target['target'].to_numpy(),
    #                sample_weight=train_input['weights'].to_numpy(),
    #                epochs=5,
    #                batch_size=BATCHSIZE)
    #
    # chainedModel.fit([train_input[columns].to_numpy(), train_input["logPt"].to_numpy()],
    #                  train_target['adversarialTarget'].to_numpy(),
    #                  sample_weight=train_input['weights'].to_numpy(),
    #                  epochs=5,
    #                  batch_size=BATCHSIZE)

#   classifier.predict(test_input[columns].to_numpy())



#     def customLoss(y_true, y_pred):
# #        return nsl.lib.jensen_shannon_divergence(y_true, y_pred, axis=1)
#
#         return nsl.lib.pairwise_distance_wrapper(
#             y_pred,
#             y_true,
#             distance_config=nsl.configs.DistanceConfig(
#                 distance_type=nsl.configs.DistanceType.JENSEN_SHANNON_DIVERGENCE,
#                 sum_over_axis=1
#             )
#         )


#     network.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
# #    network.compile(optimizer=tf.optimizers.SGD(learning_rate=1e-3),
#                     loss=lambda y, model: -model.log_prob(y+1e-6))
# #                    loss=JensenShannonDivergence)
# #                    loss=customLoss)
# #                    loss="categorical_crossentropy")
#

    # print(network.summary())
    # network.fit(datasetTrain, epochs=20)
#    network.fit(allData_input, allData_target, sample_weight=weights, epochs=5)
#    network.fit(allData_input, allData_target, epochs=5)

    # classifier = createClassifier()
    # classifier.compile(optimizer='adam',
    #                    loss='binary_crossentropy',
    #                    metrics=['accuracy'])
    # adversary = createAdversary(1, 0)
    # adversary.compile(optimizer='adam',
    #                   loss=kerasCustomLoss(adversary.get_layer('input').input))
    #
    # classifier.fit(train_data, train_target, batch_size=BATCHSIZE, epochs=10)
    # predictions = classifier.predict(test_data[(test_target==0)])
    # adversary.fit(predictions, test_data[(test_target==0)]['TransverseMass'].values, batch_size=BATCHSIZE, epochs=100)
    # advOut = adversary.predict(predictions)
    #
    # means = np.mean(advOut, axis=0)
    # GMM = createGMM(means[:20], means[20:40], means[40:])
    # sampled = GMM.sample(10000)
    #
    #
    # sampleInput = np.random.uniform(low=0.0, high=199.0, size=TESTSET_SIZE).reshape(-1,1)
    # sampleInput_scaled = scaler1.transform(sampleInput)
    # sampled = scaler2.inverse_transform(network.predict(sampleInput_scaled))
    # sampled = np.argmax(network.predict(allData_input[:10000]), axis=1)
    # sampled = np.sum(tf.one_hot(sampled, PTBINS), axis=0)
    #
    # true = np.sum(allData_target[:10000], axis=0)
    #
    # width = binning[1]-binning[0]
    #
    # plt.bar(binning[:-1]+width/2, sampled, width=width, label="GMM", alpha=0.7, linewidth=1.0, edgecolor='black', color='green')
    # plt.bar(binning[:-1]+width/2, true, width=width, label="True", alpha=0.7, linewidth=1.0, edgecolor='black', color='blue')
    # plt.legend()
    # plt.savefig("plots/outputHistograms.pdf")

    #sampled = network.predict(sampleInput_scaled)

    # print(sampled)
    #
    # plt.scatter(sampleInput, sampled)
    # plt.ylim(0.0, 200.0)
    # plt.xlim(0.0, 200.0)
    # plt.savefig("plots/sampledScatter.pdf")
#     print(sampled.shape)
#     print(sampled)
#
#     n, _ = np.histogram(sampleInput, bins=binning)
#     toScale, _ = np.histogram(sampleInput, bins=binning, weights=sampled.flatten())
#     resultBinContent = toScale/n
#
#     plt.bar(binning[:-1]+width/2, resultBinContent, width=width, label="GMM", alpha=0.7, linewidth=1.0, edgecolor='black', color='green')
#     plt.bar(binning[:-1]+width/2, binContent, width=width, label="True", alpha=0.7, linewidth=1.0, edgecolor='black', color='blue')
#     plt.legend()
#     plt.savefig('plots/TestOutput.pdf')




if __name__ == "__main__":
    main()