from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow as tf
from hyperOptimization import HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, HP_ACTIVATION, BATCHSIZE, PTBINS
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

def JensenShannonDivergence(y_true, y_pred):
    k = tf.keras.losses.KLDivergence()
    m = (y_true + y_pred)/2.0
    return tf.reduce_sum((k(y_true, m) + k(y_pred, m))/2.0, axis=0)

def createChainedModel(classifier, adversary):
    return keras.Model(input=classifier.input, output=adversary(classifier.output))

def createAdversary():
    event_shape = [1]
    numberOfGaussians = 20


    params_size = tfp.layers.MixtureSameFamily.params_size(numberOfGaussians,
                                                              component_params_size=tfp.layers.IndependentNormal.params_size(event_shape))
    model = tf.keras.Sequential([
        keras.Input(shape=(1), name='input'),
        keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform', name='hidden'),
        keras.layers.Dense(params_size, activation=None),
#        tfp.layers.MixtureSameFamily(numberOfGaussians, tfp.layers.IndependentNormal(event_shape)),
        tfp.layers.MixtureNormal(numberOfGaussians, event_shape),
#        keras.layers.Dense(PTBINS, activation=tf.nn.softmax, kernel_initializer='glorot_uniform')
    ])
    return model


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
