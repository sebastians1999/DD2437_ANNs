import numpy as np
from helper import *
from typing import Optional


class RBFNetwork: 

    def __init__(self, centers: Optional[np.ndarray] = None, sigma: Optional[float] = None):
        self.centers=centers
        self.sigma=sigma
        self.weights=None

    def set_sigma(self,sigma):
        self.sigma = sigma

    def fit_rbf_units(self, X, num_units, epochs, eta:float = 0.01, use_soft_competition:bool= False, k_winners:int = 3):


        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X = np.atleast_2d(X)

        #X = np.transpose(X). -> changed for last task.
        N, D = X.shape

        min_val = np.min(X)
        max_val = np.max(X)

        # print(min_val)
        # print(max_val)

        np.random.seed(42)

        self.centers = np.random.uniform(min_val, max_val, (num_units, D))

        # unit initialisation to provoke dead units
        # self.centers = np.zeros((num_units, D))
        # cluster_center = (max_val + min_val) / 2  
        # cluster_size = (max_val - min_val) / 10   

        # # 50% clustered around the center
        # num_clustered = num_units // 2
        # self.centers[:num_clustered] = np.random.uniform(
        #     cluster_center - cluster_size / 2, 
        #     cluster_center + cluster_size / 2, 
        #     (num_clustered, D)
        # )
        # the rest is spread across the full range
        # self.centers[num_clustered:] = np.random.uniform(min_val, max_val, (num_units - num_clustered, D))
        
        #print(f"Initial centers:\n{self.centers}")

        win_counts = np.zeros(num_units)
        win_counts_final = np.zeros(num_units)
        total_distances_per_unit = np.zeros(num_units) 
        sample_assignments = np.zeros(N, dtype=int)

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(N)
            shuffled_X = X[shuffled_indices, :]

            for i,sample in enumerate(shuffled_X):
                distances = np.linalg.norm(self.centers - sample, axis=1) #compute euclidean distance

                if use_soft_competition:

                    winner_idx = np.argsort(distances)[:k_winners]
                    weights = 1.0 / (distances[winner_idx] + 1e-10) # take inverse e.g. closer units get a higher weight update
                    weights = weights / np.sum(weights)
                    for idx, w in zip(winner_idx, weights):
                        win_counts[idx] += w  
                        self.centers[idx] += eta * w * (sample - self.centers[idx])

                else: 
                
                    winner_idx = np.argmin(distances) #find the closest distance
                    winner_distance = distances[winner_idx]
                    win_counts[winner_idx] += 1
                    self.centers[winner_idx] += eta * (sample - self.centers[winner_idx]) #update weights

               
                if epoch == epochs - 1: #only for last epoch accumulate distances
                    winner_idx = np.argmin(distances)
                    total_distances_per_unit += distances
                    win_counts_final[winner_idx] += 1
                    sample_assignments[shuffled_indices[i]] = winner_idx


            #print(f"Epoch {epoch}: Avg win count: {win_counts.mean()}")

        print(f"Final wins per Unit: {win_counts_final}")
        avg_quantization_error = np.sum(total_distances_per_unit) / (N * num_units) #compute average quantization error
        print(f"Average Quantization Error (AQE): {avg_quantization_error}")

        avg_distance_per_unit = total_distances_per_unit / N  #average per unit over samples, helps to identify dead units
        print(f"Average Distance per Unit: {avg_distance_per_unit}") 

        self.widths = np.zeros(num_units) 
        
        for j in range(num_units):
            assigned_samples = X[sample_assignments == j]  #sample from this unit

            if len(assigned_samples) > 0:  #avoid division by zero (essentially dead units)
                distances_to_centroid = np.linalg.norm(assigned_samples - self.centers[j], axis=1)
                self.widths[j] = np.std(distances_to_centroid)  #standard deviation as width
            else:
                self.widths[j] = np.mean(avg_distance_per_unit)  # default to global width

        print(f"Computed widths (sigma): {self.widths}")
        print("Final centers:\n", self.centers)

        self.sigma = np.mean(self.widths) #setting width 


        return self.centers, self.widths, avg_quantization_error


    def gaussian(self, x, c): 

        return np.exp(-np.linalg.norm(x - c)**2 / (2 * self.sigma**2))
    
    def phi_matrix(self, X):

        #print(f"Shape of X before phi matrix transformation: {X.shape}")
        phi_matrix = np.zeros((len(X), len(self.centers)))

        for i, x in enumerate(X):

            for e, c in enumerate(self.centers):

                phi_matrix[i][e]= self.gaussian(x,c)

        return phi_matrix
    

    def fit(self, X, y):

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        phi = self.phi_matrix(X)
        self.weights = np.linalg.pinv(phi) @ y

    def fit_online(self, X_train, y_train,X_test, y_test, eta, num_epoch:int = None):

        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1) #ensures compability with 1D array
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1) #ensures compability with 1D array
        phi = self.phi_matrix(X_train)

        output_dim = y_train.shape[1] 
        self.weights = np.random.normal(loc=0.0, scale=0.1, size=(phi.shape[1], output_dim))


        epoch_reached = False
        epoch_count = 0
        train_mae_history = []
        test_mae_history = []


        train_mae_history.append(mse(targets=y_train, predicted=self.predict(X_train))) #error at epoch 0

        while not epoch_reached: 

            #according to lab shuffle data new for each epoch
            X_shuffled, y_shuffled = shuffle_data(x=X_train, y=y_train)

            for i, sample in enumerate(X_shuffled):
                
                sample_arr = np.array([sample]) #needs to be an array in order that it runs with phi_matrix
                phi_vector = self.phi_matrix(sample_arr)[0]
                y_pred = self.predict(sample_arr)[0]
                error = y_shuffled[i] - y_pred
                self.weights += eta * np.outer(phi_vector, error) 

            train_mae_history.append(mae(y_true=y_train, y_pred=self.predict(X_train)))
            test_mae_history.append(mae(y_true=y_test, y_pred=self.predict(X_test)))

            epoch_count += 1

            if epoch_count == num_epoch:
                epoch_reached = True

        return  train_mae_history, test_mae_history


    def predict(self, X):
        phi = self.phi_matrix(X)
        return phi@self.weights

