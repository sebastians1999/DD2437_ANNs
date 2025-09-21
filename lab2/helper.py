import numpy as np

def mse(targets, predicted):
       e = targets - predicted
       return 0.5 * np.mean(e**2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def shuffle_data(x, y):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    return x[indices], y[indices]