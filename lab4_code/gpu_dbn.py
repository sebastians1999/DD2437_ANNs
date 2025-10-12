from gpu_util import *
from gpu_rbm import RestrictedBoltzmannMachine
if USE_GPU:
    import numpy as np


#This is a straight copy from dbn.py


class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        self.n_labels = n_labels
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        
        vis = true_img # visible layer gets the image data
        
        lbl = xp.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
        
        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.
        # 
        #Our implementation with batch processing
        predicted_lbl = xp.zeros(true_lbl.shape)
        
        for batch_start in range(0,n_samples,self.batch_size):
            
            batch_end = min(batch_start+self.batch_size,n_samples)
            batch_size = batch_start - batch_end
            
            vis_batch = vis[batch_start:batch_end]
            lbl_batch = lbl[batch_start:batch_end]
            
            #through pass calculations
            _, hid_1_batch = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis_batch)
            _, pen_batch = self.rbm_stack['hid--pen'].get_h_given_v_dir(hid_1_batch)
                    
            pen_lbl_batch = xp.concatenate([pen_batch, lbl_batch], axis=1)
            
            # gipps sampling for to top RBM hidden layer
            for _ in range(self.n_gibbs_recog):
    
                _, top_batch = self.rbm_stack['pen+lbl--top'].get_h_given_v(pen_lbl_batch)
                    
                _, pen_lbl_batch = self.rbm_stack['pen+lbl--top'].get_v_given_h(top_batch)
                
            predicted_lbl[batch_start:batch_end] = pen_lbl_batch[:, -self.n_labels:]
            
        print ("accuracy = %.2f%%"%(100.*xp.mean(xp.argmax(predicted_lbl,axis=1)==xp.argmax(true_lbl,axis=1))))
    
        return
        

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).
 
        # according to slides (or what can be derived) initialised by sampling from bias of visible layer, other methods are possible for this too
        pen_bias = self.rbm_stack['pen+lbl--top'].bias_v[:-self.n_labels]
        probs_pen_bias = sigmoid(pen_bias)
        pen_initial = xp.random.binomial(n=1, p=probs_pen_bias, size=(n_sample, self.sizes["pen"]))
        
        for _ in range(self.n_gibbs_gener):
            
            
            pen_lbl = xp.concatenate([pen_initial,lbl],axis=1)
            
            # calculate hidden layer of last rbm
            _, top = self.rbm_stack['pen+lbl--top'].get_h_given_v(pen_lbl)
             
            # calculate visible layer of last rbm   
            _, pen_lbl = self.rbm_stack['pen+lbl--top'].get_v_given_h(top)
            
            #from here downward pass through the network              
            pen = pen_lbl[:, :self.sizes["pen"]]     
            _, hid = self.rbm_stack['hid--pen'].get_v_given_h_dir(pen)         
            _, vis = self.rbm_stack['vis--hid'].get_v_given_h_dir(hid)

            if USE_GPU:
                vis = xp.asnumpy(vis)
            records.append( [ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
            
        anim = stitch_video(fig,records).save("%s.generate%d.mp4"%(name,xp.argmax(true_lbl)))            
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations, epochs=20):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily
        
            print ("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """
            self.rbm_stack['vis--hid'].cd1(visible_trainset=vis_trainset, n_iterations=n_iterations, epochs=epochs, file_prefix="vis--hid")
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")

            print ("training hid--pen")
            """ 
            CD-1 training for hid--pen 
            """ 
            self.rbm_stack["vis--hid"].untwine_weights()
            probs_h1, visible_h1 = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis_trainset)
            self.rbm_stack['hid--pen'].cd1(visible_trainset=probs_h1, n_iterations=n_iterations, epochs=epochs, file_prefix="hid--pen")                   
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")            

            print ("training pen+lbl--top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            self.rbm_stack["hid--pen"].untwine_weights()
            probs_h2, visible_h2 = self.rbm_stack['hid--pen'].get_h_given_v_dir(visible_h1)
            visible_top = xp.concatenate([probs_h2, lbl_trainset], axis=1)
            # if USE_GPU:
            #     visible_top = xp.array(np.block([xp.asnumpy(probs_h2), xp.asnumpy(lbl_trainset)]))
            # else:
            #     visible_top = xp.block([probs_h2, lbl_trainset])
            
            self.rbm_stack['pen+lbl--top'].cd1(visible_trainset=visible_top, n_iterations=n_iterations, epochs=epochs, file_prefix="pen+lbl--top")
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):            
                                                
                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.

                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.

                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.
                
                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = xp.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = xp.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = xp.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        xp.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        xp.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        xp.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = xp.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = xp.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = xp.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = xp.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        xp.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        xp.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        xp.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        xp.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
