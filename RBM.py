import math
import numpy as np
from scipy.special import expit   # sigmoid
import matplotlib.pyplot as plt
class RBM():
    def __init__(self, visibleLayers=784, hiddenLayers=100):
        self.visibleLayers = visibleLayers
        self.visibleLayers = visibleLayers
        # Parameters
        self.vhW = 0.1 * np.random.randn(visibleLayers, hiddenLayers)
        self.vlbias = np.zeros(visibleLayers)
        self.hlbias = -4.0 * np.ones(hiddenLayers)
        # Gradients
        self.vhW_delta = np.zeros(self.vhW.shape) # W_gradient
        self.vb_delta = np.zeros(visibleLayers) # visible unit bias gradient
        self.hb_delta = np.zeros(hiddenLayers) # hidden unit bias gradient
        
    def posetivePhase(self, visibleLayer):
        pdH = expit(np.matmul(visibleLayer, self.vhW) + self.hlbias) # probability distribution of the hidden layer.
        return (pdH, np.random.binomial(1, p=pdH))
    
    def negativePhase(self, hiddenLayer):
        pdV = expit(np.matmul(hiddenLayer, self.vhW.T) + self.vlbias)  # probability distribution of the visible layer.
        return (pdV, np.random.binomial(1, p=pdV))
    
    def compute_error_and_grads(self, batch):
        batchSize = batch.shape[0]
        v0 = batch.reshape(batchSize, -1)
        
        # Compute gradients - Positive Phase
        ph0, h0 = self.posetivePhase(v0)
        vhW_delta = np.matmul(v0.T, ph0)
        vb_delta = np.sum(v0, axis=0)
        hb_delta = np.sum(ph0, axis=0)
        
        # Compute gradients - Negative Phase
        
        # only contrastive with k = 1, i.e., method="cd"

        pv1, v1 = self.negativePhase(h0)
        ph1, h1 = self.posetivePhase(pv1)
        
        vhW_delta -= np.matmul(pv1.T, ph1)
        vb_delta -= np.sum(pv1, axis=0)
        hb_delta -= np.sum(ph1, axis=0)
        
        self.vhW_delta = vhW_delta/batchSize
        self.hb_delta = hb_delta/batchSize
        self.vb_delta = vb_delta/batchSize
        
        recon_err = np.mean(np.sum((v0 - pv1)**2, axis=1), axis=0) # sum of squared error averaged over the batch
        return recon_err
    
    def update_params(self, eta):
        self.vhW += (eta * self.vhW_delta)
        self.vlbias +=(eta * self.vb_delta)
        self.hlbias += (eta * self.hb_delta)
        
    def plot_weights(self, weight, savefile=""):
        plt.clf()
        fig, axes = plt.subplots(10, 10, gridspec_kw = {'wspace':0.1, 'hspace':0.1}, figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                axes[i, j].imshow(weight[:,i*10+j].reshape(28, 28), cmap='gray')
                axes[i, j].axis('off')
        plt.savefig(savefile)
    def reconstruct(self, V):
        Hp, Hs = self.posetivePhase(V)
        Vp, Vs = self.negativePhase(Hs)  # reconstructionPhase
        return Vp,Hs
    