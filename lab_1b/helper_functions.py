import numpy as np
import matplotlib.pyplot as plt


def mse(targets, predicted):
       e = targets - predicted
       return 0.5 * np.mean(e**2)

def accuracy(targets, predicted):
    pred = np.where(predicted>0, 1, 0)
    true = np.where(targets>0, 1, 0)
    acc = np.count_nonzero(pred == true) / targets.size
    return acc

def plot_learning_curve(epochs, train_mse, train_ratios, val_mse=None, val_ratios=None, function_approx: bool = False):
    


    if function_approx:
        plt.figure(figsize=(10,6))
        plt.plot(epochs, train_mse, color="red", label="MSE (Train)")

        if val_mse:

            plt.plot(epochs, val_mse, color="red", label="MSE (Val)", linestyle='--')

        plt.title(f'MSE every {len(epochs)} epochs')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.grid(visible=True)
        plt.legend()
        plt.show()

    else:
        plt.figure(figsize=(10,6))      
        plt.plot(epochs, train_mse, color="red", label="MSE (Train)")
        plt.plot(epochs, train_ratios, color="blue", label="Correct class ratio (Train)")

        if val_mse:
            plt.plot(epochs, val_mse, color="red", label="MSE (Val)", linestyle='--')
            plt.plot(epochs, val_ratios, color="blue", label="Correct class ratio (Val)", linestyle='--')

        plt.title(f'MSE and correct classification ratio every {len(epochs)} epochs')
        plt.xlabel('epoch')
        plt.ylabel('MSE and correct class ratio')
        plt.grid(visible=True)
        plt.legend()
        plt.show()



def visualize_function_approx(predicted): 

    x = np.arange(-5.0,5.0,0.5)
    y = np.arange(-5.0,5.0,0.5)
    xx, yy = np.meshgrid(x,y)
    zz = predicted.reshape(x.shape[0],y.shape[0])


    fig_3d = plt.figure(figsize=(10,6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='viridis')
    plt.draw()