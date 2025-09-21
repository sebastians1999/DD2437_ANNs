import numpy as np


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

    def fit_online(self, X, eta, epoch:int = None):

        phi = self.phi_matrix(X)

        



    def predict(self, X):
        phi = self.phi_matrix(X)
        return phi@self.weights



def delta_rule_online(epoch, x, y, eta = 0.001, create_plot: bool = False):

    # Weights should be initialized with small random numbers from normal distribution with zero mean
    w = np.random.normal(loc=0.0, scale=1.0, size=(1,3))
    epoch_reached = False
    epoch_count = 0
    w_history = []
    error_loss_history = []

    error_loss_history.append(mse(x=x,y=y,w=w)) #error at epoch 0

    while not epoch_reached: 

        for i in range(len(x[0])):

            w_delta = -eta*x[:,i] * (w @ x[:,i] - y[i])
            w = w + w_delta
            w_history.append(w.copy())

            if create_plot: 
                plot_classifier(ClassA=classA, ClassB=classB, w=w)

        error_loss_history.append(mse(x=x,y=y,w=w))
        epoch_count +=1

        if epoch_count == epoch: 
            epoch_reached = True

    return w, w_history, error_loss_history