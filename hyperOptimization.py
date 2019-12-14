import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np

HP_NUM_HIDDEN_LAYERS = hp.HParam("num_hidden_layers", hp.IntInterval(4, 12))
HP_NUM_UNITS = hp.HParam("num_units", hp.IntInterval(16, 64))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.1, 0.4)) # 0.3))
HP_OPTIMIZER = hp.HParam("opimizer", hp.Discrete(['adam'])) #, 'sgd']))
HP_ACTIVATION = hp.HParam("activation", hp.Discrete(['elu', 'selu']))#['relu', 'elu', 'selu']))
METRIC_ACCURACY = "accuracy"
BATCHSIZE = 256 #1024

PTBINS = 20
PTMIN = 0.0
PTMAX = 200.0
TESTSET_SIZE = 10000

PRETRAINEPOCHS=15

COLUMNS = ["MET", "tauPt", "ldgTrkPtFrac", "deltaPhiTauMet", "deltaPhiTauBjet", "bjetPt", "deltaPhiBjetMet", "TransverseMass"]
#COLUMNS= ["tauPt", "TransverseMass"]

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
