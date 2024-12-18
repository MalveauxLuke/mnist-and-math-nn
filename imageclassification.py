import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

from activation import SoftMax
from sequential import Sequential

# load model weights
with open('models/neural_network.pkl', 'rb') as f:
    loaded_network = pickle.load(f)
# create new model and initialize weights and layers with the trained model
model=Sequential()
for layer in loaded_network:
    model.add(layer)
model.softmax(SoftMax())
print("Loaded Neural Network:")

# Converts custom handwritten digits pictures into grayscale, white on black and reshapes
img = cv2.imread("handwritten_digits/one.jpg", cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (28, 28))
img_inverted = cv2.bitwise_not(img_resized)
# Normalize picture colors
img_normalized = img_inverted / 255.0
img_input = img_normalized.reshape(1, 28*28)

# predict number using model
y_pred = model.prediction(img_input)
predicted_class = np.argmax(y_pred)

print(f"Predicted: {predicted_class}")

# display picture as our model "sees" it
plt.title("Resized 28x28 Image")
plt.imshow(img_inverted, cmap="gray")
plt.axis("off")
plt.show()
