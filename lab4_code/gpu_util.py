
USE_GPU = True

if USE_GPU:
  import cupy as xp
  xp.cuda.runtime.getDeviceCount()
else:
  import numpy as xp

import matplotlib.pyplot as plt

def sigmoid(support):
    
    """ 
    Sigmoid activation function that finds probabilities to turn ON each unit. 
        
    Args:
      support: shape is (size of mini-batch, size of layer)      
    Returns:
      on_probabilities: shape is (size of mini-batch, size of layer)      
    """
    
    on_probabilities = 1./(1.+xp.exp(-support))
    return on_probabilities

def softmax(support):

    """ 
    Softmax activation function that finds probabilities of each category
        
    Args:
      support: shape is (size of mini-batch, number of categories)      
    Returns:
      probabilities: shape is (size of mini-batch, number of categories)      
    """

    expsup = xp.exp(support-xp.max(support,axis=1)[:,None])
    return expsup / xp.sum(expsup,axis=1)[:,None]

def sample_binary(on_probabilities):    

    """ 
    Sample activations ON=1 (OFF=0) from probabilities sigmoid probabilities
        
    Args:
      support: shape is (size of mini-batch, size of layer)      
    Returns:
      activations: shape is (size of mini-batch, size of layer)      
    """

    activations = 1. * ( on_probabilities >= xp.random.random_sample(size=on_probabilities.shape) )
    return activations

def sample_categorical(probabilities):

    """ 
    Sample one-hot activations from categorical probabilities
        
    Args:
      support: shape is (size of mini-batch, number of categories)      
    Returns:
      activations: shape is (size of mini-batch, number of categories)      
    """
    
    cumsum = xp.cumsum(probabilities,axis=1)
    rand = xp.random.random_sample(size=probabilities.shape[0])[:,None]    
    activations = xp.zeros(probabilities.shape)
    activations[range(probabilities.shape[0]),xp.argmax((cumsum >= rand),axis=1)] = 1
    return activations

def load_idxfile(filename):

    """
    Load idx file format. For more information : http://yann.lecun.com/exdb/mnist/ 
    """
    import struct
        
    with open(filename,'rb') as _file:
        if ord(_file.read(1)) != 0 or ord(_file.read(1)) != 0 :
           raise Exception('Invalid idx file: unexpected magic number!')
        dtype,ndim = ord(_file.read(1)),ord(_file.read(1))
        shape = [struct.unpack(">I", _file.read(4))[0] for _ in range(ndim)]
        data = xp.fromfile(_file, dtype=xp.dtype(xp.uint8).newbyteorder('>')).reshape(shape)
    return data
    
def read_mnist(dim=[28,28],n_train=60000,n_test=1000):

    """
    Read mnist train and test data. Images are normalized to be in range [0,1]. Labels are one-hot coded.
    """    

    train_imgs = load_idxfile("train-images-idx3-ubyte")
    train_imgs = train_imgs / 255.
    train_imgs = train_imgs.reshape(-1,dim[0]*dim[1])

    train_lbls = load_idxfile("train-labels-idx1-ubyte")
    train_lbls_1hot = xp.zeros((len(train_lbls),10),dtype=xp.float32)
    train_lbls_1hot[range(len(train_lbls)),train_lbls] = 1.

    test_imgs = load_idxfile("t10k-images-idx3-ubyte")
    test_imgs = test_imgs / 255.
    test_imgs = test_imgs.reshape(-1,dim[0]*dim[1])

    test_lbls = load_idxfile("t10k-labels-idx1-ubyte")
    test_lbls_1hot = xp.zeros((len(test_lbls),10),dtype=xp.float32)
    test_lbls_1hot[range(len(test_lbls)),test_lbls] = 1.

    return train_imgs[:n_train],train_lbls_1hot[:n_train],test_imgs[:n_test],test_lbls_1hot[:n_test]

def viz_rf(weights, it, grid, file_prefix=None):

    """
    Visualize receptive fields and save 
    """
    if USE_GPU:
        weights = xp.asnumpy(weights)
    fig, axs = plt.subplots(grid[0],grid[1],figsize=(grid[1],grid[0]))#,constrained_layout=True)
    plt.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0,hspace=0)        
    imax = abs(weights).max()
    for x in range(grid[0]):
        for y in range(grid[1]):
            axs[x,y].set_xticks([]);
            axs[x,y].set_yticks([]);
            axs[x,y].imshow(weights[:,:,y+grid[1]*x], cmap="bwr", vmin=-imax, vmax=imax, interpolation=None)
    
    if file_prefix:
      plt.savefig(f"{file_prefix}_rf.iter{it:06d}.png")
    else:
      plt.savefig(f"rf.iter{it:06d}.png")
    plt.close('all')

# 2025-10-10 Ivan - Newly added function, to plot histogram
def plot_histograms(weights, it, file_prefix=None):
    if USE_GPU:
        weights = xp.asnumpy(weights)
    plt.hist(weights.flatten(), bins=50)
    plt.title(f"Weights distribution epoch {it}")
    if file_prefix:
      plt.savefig(f"{file_prefix}_rf.hist.iter{it:06d}.png")
    else:
      plt.savefig(f"rf.hist.iter{it:06d}.png")
    plt.close('all')

# 2025-10-11 Ivan - Newly added function, to plot reconstruction loss
def plot_reconstruction_loss(reconstruction_loss, file_prefix=None):
    recon_list = []
    if USE_GPU:
        for r in reconstruction_loss:
           recon_list.append(xp.asnumpy(r))
    else:
        recon_list = reconstruction_loss

    epochs = len(recon_list)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, recon_list[0], marker='o', color='blue', label='Hidden units = 500')
    plt.plot(epochs, recon_list[1], marker='o', color='red', label='Hidden units = 200')
    plt.title(f"Reconstruction loss comparison per epoch")
    if file_prefix:
      plt.savefig(f"{file_prefix}_rf.reconstruction_loss.png")
    else:
      plt.savefig(f"rf.reconstruction_loss.png")
    plt.close('all')
    

def stitch_video(fig,imgs):
    """
    Stitches a list of images and returns a animation object
    """
    import matplotlib.animation as animation
    
    return animation.ArtistAnimation(fig, imgs, interval=100, blit=True, repeat=False)    
