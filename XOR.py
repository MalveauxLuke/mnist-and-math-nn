import numpy as np

from activation import ActivationLayer,SoftMax
from DenseLayer import DenseLayer
from losses import cross_entropy, cross_entropy_prime
from network import train
from sequential import Sequential
def to_one_hot(predictions, num_classes):
    one_hot = np.zeros((predictions.size, num_classes))
    one_hot[np.arange(predictions.size), predictions] = 1
    return one_hot
# XOR dataset

x_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


y_train = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

# build the neural network
model = Sequential()
model.add(DenseLayer(2, 4))
model.add(ActivationLayer("relu"))
model.add(DenseLayer(4, 2))
model.softmax(SoftMax())

# train the model
train( model=model,
    loss=cross_entropy,
    loss_prime=cross_entropy_prime,
    x_train=x_train,
    y_train=y_train,
    epochs=10000,
    learning_rate=0.1,verbose=True)


# test the model
print("Testing the model...")
for x, y_true in zip(x_train, y_train):
    x = x.reshape(1, -1)
    y_pred = model.prediction(x)
    predictions = np.argmax(y_pred, axis=1)
    num_classes = y_pred.shape[1]
    one_hot_output = to_one_hot(predictions, num_classes)
    print(f"Input: {x} -> Predicted: {one_hot_output}, True: {y_true}, Probability: {y_pred}")

