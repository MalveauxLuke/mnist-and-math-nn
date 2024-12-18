import tensorflow as tf
import pickle

from activation import ActivationLayer, SoftMax
from DenseLayer import DenseLayer
from losses import cross_entropy, cross_entropy_prime
from network import train
from sequential import Sequential
from preprocessing_mnist import preprocess_data

# load mnist dataset from tensor flow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# pre process to normalize reshape and shuffle data
x_train, y_train = preprocess_data(x_train, y_train, 60000)

model = Sequential()

# architecture can vary if necessary
model.add(DenseLayer(input_size=784, num_units=128))
model.add(ActivationLayer("relu"))
model.add(DenseLayer(input_size=128, num_units=64))
model.add(ActivationLayer("relu"))
model.add(DenseLayer(input_size=64, num_units=10))
model.softmax(SoftMax())
# train
train(model=model,
      loss=cross_entropy,
      loss_prime=cross_entropy_prime,
      x_train=x_train,
      y_train=y_train,
      epochs=200,
      learning_rate=0.01, verbose=True, batch_size=128)
# saves trained model to file
with open('models/neural_network.pkl', 'wb') as f:
    pickle.dump(model.layers, f)
