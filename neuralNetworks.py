from tensorflow import keras
from tensorflow.keras.activations import sigmoid
import tensorflow_probability as tfp
import tensorflow as tf
from hyperOptimization import HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, HP_ACTIVATION, BATCHSIZE, PTBINS, COLUMNS
import math
import numpy as np


#Performs the work of the sklearn StandardScaler. Requires the feature means and scales as input, but
#after that the values are store here for the easy deployment of the model.
#Is meant to be deployed right after the Input layer

### The means and scale input variables to the layer can be gotten from sklearn StandardScaler properties .means_ and ._scale
class StandardScalerLayer(tf.keras.layers.Layer):
    def __init__(self, means, scale, units=BATCHSIZE, **kwargs):
        self.units = units
        self.means = tf.convert_to_tensor(np.reshape(means, (1, means.shape[-1])), dtype='float32')
        self.invertedScale = tf.convert_to_tensor(1.0 / np.reshape(scale, (1, scale.shape[-1])), dtype='float32')
        super(StandardScalerLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(StandardScalerLayer, self).build(input_shape)

    def call(self, input):
        return tf.math.multiply((input-self.means), self.invertedScale)

    def get_config(self):
        return {'units': self.units}

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


def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

beta = tf.Variable(initial_value=1.0, trainable=True, name='swish_beta')
def altSwish(x):
    return x * tf.nn.sigmoid(beta*x)

def createClassifier(means, scale):
    _activation = altSwish #swish #'relu'
    _initialization = 'glorot_normal'
    _regularizer = keras.regularizers.l1(0.01)
    _nodes = 32

    _inputs = keras.Input(shape=(len(COLUMNS)), name="inputClassifier")
    x = StandardScalerLayer(means, scale)(_inputs)
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization, kernel_regularizer=_regularizer, name="dense1")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization, kernel_regularizer=_regularizer, name="dense2")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization, kernel_regularizer=_regularizer, name="dense3")(x)  #
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization, kernel_regularizer=_regularizer, name="dense4")(x)  #
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(_nodes, activation=_activation, kernel_initializer=_initialization, kernel_regularizer=_regularizer, name="dense5")(x)  #
    x = keras.layers.BatchNormalization()(x)
    _outputs = keras.layers.Dense(1, activation="sigmoid", kernel_initializer=_initialization, kernel_regularizer=_regularizer, name="dense_output")(x)

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
    numberOfGaussians = 10


    params_size = tfp.layers.MixtureSameFamily.params_size(numberOfGaussians,
                                                              component_params_size=tfp.layers.IndependentNormal.params_size(event_shape))

    inputs = keras.Input(shape=(1), name="inputAdversary")
    auxiliary = keras.Input(shape=(1), name="auxiliaryAdversary")
    x = keras.layers.Concatenate()([inputs, auxiliary])
    x = keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_uniform', name='hidden')(x)
    x = keras.layers.Dense(params_size, activation=None)(x)
    out = tfp.layers.MixtureNormal(numberOfGaussians, event_shape, name="out_adversary")(x)

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

def setTrainable(model, isTrainable):
    model.trainable = isTrainable
    for layer in model.layers:
        layer.trainable = isTrainable

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

