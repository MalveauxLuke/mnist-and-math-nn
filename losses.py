import numpy as np
# Loss functions. Will add more in future
# Model is currently specialized for mnist/softmax

def cross_entropy(y_true, y_logits):
    # y_logits are raw logits aka linear output
    # subtracting from max for stability and to avoid overflow
    logits_exp = np.exp(y_logits - np.max(y_logits, axis=-1, keepdims=True))
    softmax_probs = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)

    #epsilon = 1e-9
    #softmax_probs_clipped = np.clip(softmax_probs, epsilon, 1 - epsilon)

    log_softmax=np.log(softmax_probs)
    loss = -np.sum(y_true * log_softmax)
    return np.mean(loss)
def cross_entropy_prime(y_true, y_logits):
    # again subtracting max for numerical stability
    logits_exp = np.exp(y_logits - np.max(y_logits, axis=-1, keepdims=True))
    softmax_probs = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)
    # because of the design of softmax derrivative is very simple
    gradient = (softmax_probs-y_true) / y_true.shape[0]
    return gradient
