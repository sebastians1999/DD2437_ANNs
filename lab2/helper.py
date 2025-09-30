import numpy as np

def mse(targets, predicted):
       e = targets - predicted
       return 0.5 * np.mean(e**2)

# def mae(y_true, y_pred):
#     return np.mean(np.abs(y_true - y_pred))


# needs to be proper implemented
def mae(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)  
    elif y_true.ndim > 2:
        raise ValueError("y_true must be 1D or 2D array")
    
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    elif y_pred.ndim > 2:
        raise ValueError("y_pred must be 1D or 2D array")
    
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Number of samples mismatch: y_true {y_true.shape[0]} vs y_pred {y_pred.shape[0]}")
    if y_true.shape[1] != y_pred.shape[1]:
        y_true = np.tile(y_true, (1, y_pred.shape[1] // y_true.shape[1])) 

    return np.mean(np.abs(y_true - y_pred))

def shuffle_data(x = None, y = None):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    return x[indices], y[indices]