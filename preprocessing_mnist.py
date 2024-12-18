import numpy as np
# used to preprocess data.

# one_hot encoding for correct categorical values
def manual_to_categorical(y, num_classes=10):
    n_samples = len(y)
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot
# preprocess and limit data
def preprocess_data(x, y, limit, shuffle=False):
    x = x.reshape(x.shape[0], 28 * 28)
    x = x.astype("float32") / 255
    y = manual_to_categorical(y)
    x, y = x[:limit], y[:limit]
    if shuffle:
        indices = np.arange(x)
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
    return x, y
