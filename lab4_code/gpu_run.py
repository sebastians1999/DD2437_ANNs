from gpu_util import *
from gpu_rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
import time

if __name__ == "__main__":
    print("Backend:", xp.__name__)

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000) # Ivan 2025-10-10: Full dataset -> n_train=60000, n_test=10000. Startr with samller values

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")
    reconstruction_loss_list = []
    
    # 2025-10-10 Ivan: Duplicate this to do 4.1.2
    ##### New RBM with Hidden nodes = 500 #####
    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=500, # Ivan 2025-10-10: Was initially 200, but both video and task 4.1 says start with 500?? So we strart with 500
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20 # Changed to 20 as task 4.1 says
    )
    # Start clock
    if USE_GPU:
        xp.cuda.Stream.null.synchronize()
    t0 = time.time()
    
    recon_loss = rbm.cd1(visible_trainset=train_imgs, n_iterations=10000, epochs=20, file_prefix="4.1.2_500hn")
    
    # Stop clock
    if USE_GPU:
        xp.cuda.Stream.null.synchronize()
    print("Elapsed:", time.time() - t0)
    reconstruction_loss_list.append(recon_loss)

    ##### New RBM with Hidden nodes = 200 #####
    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=200, # Ivan 2025-10-10: Was initially 200, but both video and task 4.1 says start with 500?? So we strart with 500
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20 # Changed to 20 as task 4.1 says
    )
    
    # Start clock
    if USE_GPU:
        xp.cuda.Stream.null.synchronize()
    t0 = time.time()
    
    recon_loss = rbm.cd1(visible_trainset=train_imgs, n_iterations=10000, epochs=20, file_prefix="4.1.2_200hn")

    # Stop clock
    if USE_GPU:
        xp.cuda.Stream.null.synchronize()
    print("Elapsed:", time.time() - t0)

    reconstruction_loss_list.append(recon_loss)

    plot_reconstruction_loss(reconstruction_loss=reconstruction_loss_list, file_prefix="4.1.2")
    
    # ''' deep- belief net '''

    # print ("\nStarting a Deep Belief Net..")
    
    # dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
    #                     image_size=image_size,
    #                     n_labels=10,
    #                     batch_size=10
    # )
    
    # ''' greedy layer-wise training '''

    # dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    # dbn.recognize(train_imgs, train_lbls)
    
    # dbn.recognize(test_imgs, test_lbls)

    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="rbms")

    # ''' fine-tune wake-sleep training '''

    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    # dbn.recognize(train_imgs, train_lbls)
    
    # dbn.recognize(test_imgs, test_lbls)
    
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
