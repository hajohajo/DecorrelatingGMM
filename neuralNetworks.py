from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow as tf
from hyperOptimization import HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, HP_ACTIVATION, BATCHSIZE
import math
import numpy as np


def createClassifier():
    _activation = 'elu'
    _initialization = 'glorot_normal'
    _nodes = 64

    _inputs = keras.Input(shape=(8), name="input")
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization)(_inputs)
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization)(x)
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization)(x)
    _outputs = keras.layers.Dense(1, activation="sigmoid", kernel_initializer=_initialization)(x)

    model = keras.Model(inputs=_inputs, outputs=_outputs)

    return model

def createAdversary(classifierOutput, auxiliaryInputs):
    KMIX = 20
    _activation = 'relu'
    _initialization = 'glorot_normal'
    _nodes = 64
    _outputs = KMIX*3

    _inputs = keras.Input(shape=(1), name='input')
    _logPt = keras.Input(shape=(1))
    x = keras.layers.Concatenate()([_inputs, _logPt])
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization)(_inputs)
    _outputsPi = keras.layers.Dense(KMIX, activation=tf.nn.softmax)(x)
    _outputsSigma = keras.layers.Dense(KMIX, activation='linear')(x)
    _outputsMu = keras.layers.Dense(KMIX, activation='linear')(x)
    _output = keras.layers.Concatenate()([_outputsPi, _outputsSigma, _outputsMu])

    model = keras.Model(inputs=[_inputs, _logPt], outputs=_output)
    return model

def createChainedModel(classifier, adversary):
    return keras.Model(input=classifier.input, output=adversary(classifier.output))

def GMMTrainer():
    event_shape = [1]
    numberOfGaussians = 5

    params_size = tfp.layers.MixtureSameFamily.params_size(numberOfGaussians,
                                                              component_params_size=tfp.layers.IndependentNormal.params_size(event_shape))
    model = tf.keras.Sequential([
        keras.Input(shape=(1), name='input'),
        keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform', name='hidden'),
        keras.layers.Dense(params_size, activation=None),
        # _pi = keras.layers.Dense(20, activation=tf.nn.softmax)(_hidden)
        # _sigma = keras.layers.Dense(20, activation='exponential')(_hidden)
        # _mu = keras.layers.Dense(20, activation='linear')(_hidden)
        tfp.layers.MixtureSameFamily(numberOfGaussians, tfp.layers.IndependentNormal(event_shape))
    ])
#    _output = keras.layers.Concatenate()([_pi, _sigma, _mu])
#    model = keras.Model(inputs=_inputs, outputs=_output)
    return model

def binOutputs(output):
#    binning = np.linspace(50, 200, 15)

    binContent = tf.histogram_fixed_width(output, tf.constant([50.0, 200.0]), 30, dtype=tf.dtypes.int64)
#    binContent, _ = np.histogram(output, bins=binning, density=True)
    return tf.dtypes.cast(binContent, dtype=tf.dtypes.float64)

def GMMTrainerLoss(y_true, y_pred):
    pi = y_pred[:, :20]
    sigma = y_pred[:, 20:40]
    mu = y_pred[:, 40:]
    gmm = createGMM(pi, sigma, mu)

    y_pred = gmm.sample(BATCHSIZE)

    binned_P = binOutputs(y_pred)
    binned_Q = binOutputs(y_true)

#    binned_M = (binned_P+binned_Q)/2.0

    binned_P = y_pred
    binned_Q = y_true
    binned_M = (binned_P+binned_Q)/2

    KL_PM = tf.keras.losses.kullback_leibler_divergence(binned_P, binned_M)
    KL_QM = tf.keras.losses.kullback_leibler_divergence(binned_Q, binned_M)
    JSD_PQ = (KL_PM+KL_QM)

    return JSD_PQ

def createGMM(pi, sigma, mu):
    mixtureModel = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=pi),
        components_distribution=tfp.distributions.Normal(
            loc=mu,
            scale=sigma
        )
    )

    return mixtureModel

def kerasCustomLoss(predictedClassification): #, realClass): #(y_true, y_pred):
    indexAbove = tf.math.greater_equal(predictedClassification, tf.broadcast_to(tf.constant(0.5), predictedClassification.shape)) # >= 0.5)]# & (realClass == 0)]
    indexBelow = tf.math.less(predictedClassification, 0.5) #[(np.array(predictedClassification) < 0.5)]# & (realClass == 0)]
    print(indexAbove)
    def realLoss(y_true, y_pred):
        y_pred_above = tf.boolean_mask(y_pred, indexAbove)
        y_pred_below = tf.boolean_mask(y_pred, indexBelow)

        out_piAbove = tf.reduce_mean(y_pred_above[:, :20], axis=0)
        out_sigmaAbove = tf.reduce_mean(y_pred_above[:, 20:40], axis=0)
        out_muAbove = tf.reduce_mean(y_pred_above[:, 40:], axis=0)

        # out_piAbove = tf.reduce_mean(y_pred[indexAbove][:, :20], axis=0)
        # out_sigmaAbove = tf.reduce_mean(y_pred[indexAbove][:, 20:40], axis=0)
        # out_muAbove = tf.reduce_mean(y_pred[indexAbove][:, 40:], axis=0)
        mixtureModelAbove = createGMM(out_piAbove, out_sigmaAbove, out_muAbove)



        # out_piBelow = tf.reduce_mean(y_pred[indexBelow][:, :20], axis=0)
        # out_sigmaBelow = tf.reduce_mean(y_pred[indexBelow][:, 20:40], axis=0)
        # out_muBelow = tf.reduce_mean(y_pred[indexBelow][:, 40:], axis=0)
        # mixtureModelBelow = createGMM(out_piBelow, out_sigmaBelow, out_muBelow)
        #
        # y_below = mixtureModelBelow.sample(BATCHSIZE)
        y_above = mixtureModelAbove.sample(BATCHSIZE)

        return 0.0

    return realLoss

gausNorm = 1.0/math.sqrt(2*math.pi)
def tf_normal(y, mu, sigma):
    result = tf.sub(y, mu)
    result = tf.mul(result, tf.inv(sigma))
    result = -tf.square(result)/2
    return tf.mul(tf.exp(result), tf.inv(sigma))*gausNorm

def getLossFunc(out_pi, out_sigma, out_mu, y):
    result = tf_normal(y, out_mu, out_sigma)
    result = tf.mul(result, out_pi)
    result = tf.reduce_sum(result, 1, keep_dims=True)
    result = -tf.log(result)
    return tf.reduce_mean(result)


def train_test_Classifier(hparams, train_data, test_data):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(8))
    for i in range(hparams[HP_NUM_HIDDEN_LAYERS]):
        model.add(tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=hparams[HP_ACTIVATION]))
        model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_data, epochs=5)
    _, accuracy = model.evaluate(test_data)
    return accuracy
