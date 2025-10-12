from gpu_util import *
from gpu_rbm import RestrictedBoltzmannMachine
if USE_GPU:
    import numpy as np
    


class StackedRBM():
    
    def __init__(self, sizes, image_size, batch_size):
        self.rbm1 = RestrictedBoltzmannMachine(ndim_visible=sizes["vis"],ndim_hidden=sizes["hid"], is_bottom=True, image_size=image_size,batch_size=batch_size)
        
        self.rbm2 = RestrictedBoltzmannMachine(ndim_visible=sizes["hid"],ndim_hidden=sizes["pen"],batch_size=batch_size)
        
        self.sizes = sizes
        self.image_size = image_size
        self.batch_size = batch_size
        
    def train_greedylayerwise(self, vis_trainset, n_iterations, epochs):
        print("training vis--hid")
        recon_loss1 = self.rbm1.cd1(visible_trainset=vis_trainset, n_iterations=n_iterations, epochs=epochs, file_prefix="stacked_rbm1")
        
        print("training hid--pen")
        self.rbm1.untwine_weights()
        _, hid_trainset = self.rbm1.get_h_given_v_dir(visible_minibatch=vis_trainset)
        recon_loss2 = self.rbm2.cd1(visible_trainset=hid_trainset, n_iterations=n_iterations, epochs=epochs, file_prefix="stacked_rbm2")
        
        return recon_loss1,recon_loss2