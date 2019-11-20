from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow as tf
from hyperOptimization import HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, HP_ACTIVATION, BATCHSIZE, PTBINS, COLUMNS
import math
import numpy as np


def trainingLoop(classifier, adversary, train_input, train_target, epochs):
    numberOfBatches = np.ceil(train_input.shape[0] / BATCHSIZE)
    inputBatches = np.array_split(train_input[COLUMNS].to_numpy(), numberOfBatches)
    targetBatches = np.array_split(train_target['target'].to_numpy(), numberOfBatches)
    adversaryTargetBatches = np.array_split(train_target["adversarialTarget"].to_numpy(), numberOfBatches)
    auxiliaryBatches = np.array_split(train_input['logPt'].to_numpy(), numberOfBatches)
    weightBatches = np.array_split(train_input["weights"], numberOfBatches)

    indices = range(len(inputBatches))
    progbar = tf.keras.utils.Progbar(len(inputBatches), verbose=1)
    for i in range(epochs):
        if (i != 0):
            print("\n\n")
        epochClassifierLosses = np.empty(int(numberOfBatches), dtype=float)
        epochAdversaryLosses = np.empty(int(numberOfBatches), dtype=float)

        for index in indices:
            classifierLoss = classifier.train_on_batch(inputBatches[index], targetBatches[index],
                                                       sample_weight=weightBatches[index])
            adversaryLoss = adversary.train_on_batch([inputBatches[index], auxiliaryBatches[index]],
                                                        [targetBatches[index], adversaryTargetBatches[index]],
                                                        sample_weight=[weightBatches[index], weightBatches[index]])
            progbar.update(index, values=[("Classifier loss", classifierLoss), ("Adversary loss", adversaryLoss[2])])


from tensorflow.keras.activations import sigmoid
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

beta = tf.Variable(initial_value=1.0, trainable=True, name='swish_beta')
def altSwish(x):
    return x * tf.nn.sigmoid(beta*x)

def createClassifier():
    _activation = altSwish #swish #'relu'
    _initialization = 'glorot_normal'
    _nodes = 16

    _inputs = keras.Input(shape=(len(COLUMNS)), name="inputClassifier")
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization)(_inputs)
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization)(x)
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization)(x)  #
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization)(x)  #
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization)(x)  #
    _outputs = keras.layers.Dense(1, activation="sigmoid", kernel_initializer=_initialization, name="outputClassifier")(x)

    model = keras.Model(inputs=_inputs, outputs=_outputs, name="Classifier")

    return model

def JensenShannonDivergence(y_true, y_pred):
    k = tf.keras.losses.KLDivergence()
    m = (y_true + y_pred)/2.0
    return tf.reduce_sum((k(y_true, m) + k(y_pred, m))/2.0, axis=0)

@tf.custom_gradient
def gradReverse(x): #, gamma=1.0):
    y = tf.identity(x)
    def custom_gradient(dy):
        return -10*dy
    return y, custom_gradient

class GradReverse(tf.keras.layers.Layer):
    def __init__(self, gamma=1.0, **kwargs):
        self.gamma=gamma
        super(GradReverse, self).__init__(**kwargs)

    def call(self, x):
        return gradReverse(x) #, self.gamma)

def createChainedModel_v3(classifier, adversary, gamma):

    x = GradReverse(gamma)(classifier.output)
    chainedModel = tf.keras.Model(inputs=[classifier.input, adversary.inputs[1]],
                                  outputs=[classifier.output, adversary([x, adversary.inputs[1]])])

    return chainedModel

def createAdversary():
    event_shape = [1]
    numberOfGaussians = 20


    params_size = tfp.layers.MixtureSameFamily.params_size(numberOfGaussians,
                                                              component_params_size=tfp.layers.IndependentNormal.params_size(event_shape))

    inputs = keras.Input(shape=(1), name="inputAdversary")
    auxiliary = keras.Input(shape=(1), name="auxiliaryAdversary")
#    x = keras.layers.Concatenate()([inputs, auxiliary])
    x = keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform', name='hidden')(inputs)#(x)
    x = keras.layers.Dense(params_size, activation=None)(x)
    out = tfp.layers.MixtureNormal(numberOfGaussians, event_shape)(x)

    model = keras.Model(inputs=[inputs, auxiliary], outputs=out, name="Adversary")

#     model = tf.keras.Sequential([
#         keras.Input(shape=(1), name='input'),
#         keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform', name='hidden'),
#         keras.layers.Dense(params_size, activation=None),
#         tfp.layers.MixtureNormal(numberOfGaussians, event_shape),
# #        keras.layers.Dense(PTBINS, activation=tf.nn.softmax, kernel_initializer='glorot_uniform')
#     ])
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
