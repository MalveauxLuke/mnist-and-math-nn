import tensorflow as tf
import matplotlib.pyplot as plt

# Display any number of dataset digits
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
fig, axes = plt.subplots(1, 15, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(x_train[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f"Label: {y_train[i]}")
plt.tight_layout()
plt.show()