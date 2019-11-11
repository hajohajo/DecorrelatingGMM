from generateDummyData import generateSamples
from utilities import createDirectories
from plotting import createDistributionComparison
from neuralNetworks import train_test_Classifier, createClassifier, createAdversary, createAdversary, JensenShannonDivergence
import pandas as pd
from tensorboard.plugins.hparams import api as hp
from hyperOptimization import HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, METRIC_ACCURACY, HP_ACTIVATION, HP_NUM_HIDDEN_LAYERS, BATCHSIZE, PTBINS, PTMIN, PTMAX
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight, compute_sample_weight

from sklearn.model_selection import train_test_split


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import neural_structured_learning as nsl

tf.random.set_seed(13)
tf.executing_eagerly()

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
    columns = ["tauPt, TransverseMass"]

    createDirectories()
    signal = generateSamples(1000, isSignal=True)
    background = generateSamples(500000, isSignal=False)
#    createDistributionComparison(signal, background, columns=columns)
    allData = background #signal.append(background, ignore_index=True)
    allData = allData.sample(frac=1.0).reset_index(drop=True)
    allData_mass = allData["TransverseMass"].copy()

    binning = np.linspace(PTMIN, PTMAX, PTBINS+1)
    # width=binning[1]-binning[0]
    digitized = np.digitize(np.clip(allData['TransverseMass'].values, PTMIN, PTMAX-1.0), bins=binning, right=False) - 1
    # binContent, _ = np.histogram(allData['TransverseMass'].values, bins=binning, density=True)
    # binContent=binContent*width
    # allData['TransverseMass'] = binContent[digitized]

    allData_target = allData['TransverseMass'].values
#    allData_target = tf.one_hot(digitized, PTBINS)
#    temp = [np.argmax(y) for y in allData_target]
    weighting = class_weight.compute_class_weight('balanced', np.unique(digitized), digitized)

    weights = compute_sample_weight('balanced', digitized)
#    print(digitized[:10])
#    print(weights[:10])

    allData_input = allData['tauPt'].values #allData[columns]

    plt.scatter(allData_input[:10000], allData_target[:10000])
    plt.xlim(0.0, 200.0)
    plt.ylim(0.0, 200.0)
    plt.savefig("plots/inputsAndTargets.pdf")
    plt.clf()

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    allData_input = scaler1.fit_transform(allData_input.reshape(-1, 1))
    allData_target = scaler2.fit_transform(allData_target.reshape(-1, 1))

#    allData = allData.sample(n=30*BATCHSIZE).reset_index(drop=True)
#    train_data, test_data, train_target, test_target = train_test_split(allData[columns], allData['target'], test_size=0.1, random_state=42)


    # datasetTrain = tf.data.Dataset.from_tensor_slices((allData_input.values, allData_target.values.reshape((-1, 1))))
    print(allData_input.shape, allData_target.shape, weights.shape)
    datasetTrain = tf.data.Dataset.from_tensor_slices((allData_input, allData_target, weights))

    datasetTest = datasetTrain.take(TESTSET_SIZE)
    datasetTest_target = allData_target[:TESTSET_SIZE]
    datasetTest_mass = allData_mass[TESTSET_SIZE]
    datasetTrain = datasetTrain.skip(TESTSET_SIZE)

    datasetTest = datasetTest.batch(BATCHSIZE)
    datasetTrain = datasetTrain.batch(BATCHSIZE)

    network = createAdversary()

    def customLoss(y_true, y_pred):
#        return nsl.lib.jensen_shannon_divergence(y_true, y_pred, axis=1)

        return nsl.lib.pairwise_distance_wrapper(
            y_pred,
            y_true,
            distance_config=nsl.configs.DistanceConfig(
                distance_type=nsl.configs.DistanceType.JENSEN_SHANNON_DIVERGENCE,
                sum_over_axis=1
            )
        )


    network.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
#    network.compile(optimizer=tf.optimizers.SGD(learning_rate=1e-3),
                    loss=lambda y, model: -model.log_prob(y+1e-6))
#                    loss=JensenShannonDivergence)
#                    loss=customLoss)
#                    loss="categorical_crossentropy")


    print(network.summary())
    network.fit(datasetTrain, epochs=20)
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

    sampleInput = np.random.uniform(low=0.0, high=199.0, size=TESTSET_SIZE).reshape(-1,1)
    sampleInput_scaled = scaler1.transform(sampleInput)
    sampled = scaler2.inverse_transform(network.predict(sampleInput_scaled))
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
    plt.scatter(sampleInput, sampled)
    plt.ylim(0.0, 200.0)
    plt.xlim(0.0, 200.0)
    plt.savefig("plots/sampledScatter.pdf")
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