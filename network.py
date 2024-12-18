import numpy as np
# used to train the model
def train(model, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=False, batch_size =32):

    error_array = np.array([])  # not being used right now. will implement graph for errors
    n_samples = len(x_train)
    for epoch in range(epochs):
        error = 0
        # shuffles x_train
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        # runs in batches in order to reduce noise especially with mnist and to improve performance.
        for start in range(0, len(x_train), batch_size):
            end = min(start + batch_size, n_samples)
            batch_x = x_train[start:end]
            batch_y = y_train[start:end]
            # forward
            output = model.forward(batch_x)
            # calculate error
            error += loss(batch_y, output)
            # calculate gradient for backprop
            gradient = loss_prime(batch_y, output)
            # backwards
            model.backwards(gradient, learning_rate)
        # mean error for that batch
        error /= (len(x_train) / batch_size)

        error_array = np.append(error_array, error)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch + 1}, loss: {error}")
    return error_array
