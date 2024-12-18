from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from activation import ActivationLayer,SoftMax
from DenseLayer import DenseLayer
from losses import cross_entropy, cross_entropy_prime
from network import train
from sequential import Sequential
# Simple iris toy dataset used for initial testing of model.
def load_iris_data():
    iris = load_iris()
    X, y = iris.data, iris.target

    # encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    # normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_iris_data()
model = Sequential()
model.add(DenseLayer(4, 10))
model.add(ActivationLayer("relu"))
model.add(DenseLayer(10, 6))
model.add(ActivationLayer("relu"))
model.add(DenseLayer(6,3))
model.softmax(SoftMax())

train(
    model=model,
    loss=cross_entropy,
    loss_prime=cross_entropy_prime,
    x_train=X_train,
    y_train=y_train,
    epochs=1000,
    learning_rate=0.1,
    verbose=True
)
print("Testing the model...")
# test model
testsize=0
correct_predictions=0
for x, y_true in zip(X_test, y_test):
    x = x.reshape(1, -1)
    y_pred = model.prediction(x)
    predictions = np.argmax(y_pred, axis=1)
    num_classes = y_pred.shape[1]
    one_hot_output = np.eye(num_classes)[predictions]
    predicted = np.squeeze(one_hot_output)
    print(f"Input: {x.flatten()} -> Predicted: {one_hot_output}, True: {y_true}, Probability: {y_pred}")
    testsize +=1
    if np.array_equal(predicted,y_true):
        correct_predictions+=1
# accuracy
accuracy = (correct_predictions/testsize)*100
print(f"accuracy: {accuracy}")

