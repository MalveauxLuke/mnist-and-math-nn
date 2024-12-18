import pickle
import numpy as np
import tensorflow as tf



from activation import SoftMax
from sequential import Sequential
from preprocessing_mnist import preprocess_data

# open neural network
with open('models/neural_network.pkl', 'rb') as f:
    loaded_network = pickle.load(f)
# Create new model using previously trained model
model=Sequential()
for layer in loaded_network:
    model.add(layer)
model.softmax(SoftMax())

print("Loaded Neural Network:")

# xtrain ytrain data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test, y_test = preprocess_data(x_test, y_test, 10000)

print("Testing the model...")
test_size = 0
correct_predictions = 0

for x, y_true in zip(x_test, y_test):
    x = x.reshape(1, -1) # reshape image so numpy works
    y_pred = model.prediction(x)
    predicted_class = np.argmax(y_pred)
    true_class = np.argmax(y_true)
    print(f"Predicted: {predicted_class}, True: {true_class}")
    test_size += 1
    if predicted_class == true_class:
        correct_predictions += 1
# display accuracy
accuracy = (correct_predictions / test_size) * 100
print(f"Accuracy: {accuracy:.2f}%")