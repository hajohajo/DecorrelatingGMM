import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

HP_NUM_HIDDEN_LAYERS = hp.HParam("num_hidden_layers", hp.IntInterval(4, 12))
HP_NUM_UNITS = hp.HParam("num_units", hp.IntInterval(16, 64))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.1, 0.4)) # 0.3))
HP_OPTIMIZER = hp.HParam("opimizer", hp.Discrete(['adam'])) #, 'sgd']))
HP_ACTIVATION = hp.HParam("activation", hp.Discrete(['elu', 'selu']))#['relu', 'elu', 'selu']))
METRIC_ACCURACY = "accuracy"
BATCHSIZE = 128#1024

PTBINS = 100 #30
PTMIN = 0.0
PTMAX = 1000.0

COLUMNS = ["MET", "tauPt", "ldgTrkPtFrac", "deltaPhiTauMet", "deltaPhiTauBjet", "bjetPt", "deltaPhiBjetMet", "TransverseMass"]
#COLUMNS= ["tauPt", "TransverseMass"]

