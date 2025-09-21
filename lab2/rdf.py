import numpy as np
from helper import *


class RBFNetwork: 

    def __init__(self, centers, sigma):
        self.centers=centers
        self.sigma=sigma
        self.weights=None

    
    def gaussian(self, x, c): 

        return np.exp(-np.linalg.norm(x - c)**2 / (2 * self.sigma**2))
    
    def phi_matrix(self, X):

        phi_matrix = np.zeros((len(X), len(self.centers)))

        for i, x in enumerate(X):

            for e, c in enumerate(self.centers):

                phi_matrix[i][e]= self.gaussian(x,c)

        return phi_matrix
    

    def fit(self, X, y):
        
        phi = self.phi_matrix(X)
        self.weights = np.linalg.pinv(phi) @ y

    def fit_online(self, X_train, y_train,X_test, y_test, eta, num_epoch:int = None):

        phi = self.phi_matrix(X_train)

        # Weights should be initialized with small random numbers from normal distribution with zero mean
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=phi.shape[1])


        epoch_reached = False
        epoch_count = 0
        train_mse_history = []
        test_mse_history = []


        train_mse_history.append(mse(targets=y_train, predicted=self.predict(X_train))) #error at epoch 0

        while not epoch_reached: 

            #according to lab shuffle data new for each epoch
            X_shuffled, y_shuffled = shuffle_data(x=X_train, y=y_train)

            for i, sample in enumerate(X_shuffled):
                
                sample_arr = np.array([sample]) #needs to be an array in order that it runs with phi_matrix
                phi_vector = self.phi_matrix(sample_arr)[0]
                y_pred = self.predict(sample_arr)[0]
                error = y_shuffled[i] - y_pred
                self.weights += eta * error * phi_vector 

            train_mse_history.append(mse(targets=y_train, predicted=self.predict(X_train)))
            test_mse_history.append(mse(targets=y_test, predicted=self.predict(X_test)))

            epoch_count += 1

            if epoch_count == num_epoch:
                epoch_reached = True

        return  train_mse_history, test_mse_history


    def predict(self, X):
        phi = self.phi_matrix(X)
        return phi@self.weights

