import math
import numpy as np
from Layer import *
from GradDescType import *
from sklearn.utils import shuffle

class Network(object):
    def __init__(self,X,Y,num_layers,drop_out = 0.0, activationF=ActivationType.SIGMOID,last_layerAF= ActivationType.SOFTMAX):
        self.X = X
        self.Y = Y
        self.num_layers = num_layers
        self.Layers = []  # network contains list of layers
        self.last_layerAF = last_layerAF
        for i in range(len(num_layers)):
            if (i == 0):  # first layer
                layer =  Layer(num_layers[i],X.shape[1],False,drop_out, activationF) 
            elif (i == len(num_layers)-1):  # last layer
                layer =  Layer(Y.shape[1],num_layers[i-1],True,drop_out, last_layerAF) 
            else:  # intermediate layers
                layer = Layer(num_layers[i],num_layers[i-1],False,drop_out, activationF) 
            self.Layers.append(layer)
         
    def forward(self,input_data):  # evaluates all layers
        out_layer = self.Layers[0].forward(input_data)
        for i in range(1,len(self.num_layers)):
            out_layer = self.Layers[i].forward(out_layer)
        return out_layer

    def Train(self, epochs,learningRate, lambda1, gradDescType, batchSize=1):
        for j in range(epochs):
            loss = 0
            self.X, self.Y = shuffle(self.X, self.Y, random_state=0)
            zz = self.X.shape[0]
            for i in range(self.X.shape[0]):
                aout = self.forward(self.X[i]) # actual output of network
                if (self.last_layerAF == ActivationType.SOFTMAX):
                    loss += -(self.Y[i] * np.log(aout)).sum()
                else:
                    loss += ((aout - self.Y[i]) * \
                                (aout - self.Y[i])).sum()
                lnum = len(self.num_layers)-1  # last layer number
                # compute deltas, grads on all layers
                while(lnum >= 0):
                    if (lnum == len(self.num_layers)-1):  # last layer
                        if (self.last_layerAF == ActivationType.SOFTMAX):
                            self.Layers[lnum].delta = -self.Y[i]+ self.Layers[lnum].a
                        else:
                            self.Layers[lnum].delta = -(self.Y[i]-self.Layers[lnum].a) * self.Layers[lnum].derivAF
                    else: # intermediate layer
                        self.Layers[lnum].delta = np.dot(self.Layers[lnum+1].W.T,self.Layers[lnum+1].delta) * self.Layers[lnum].derivAF
                    if (lnum > 0):  #previous output
                        prevOut = self.Layers[lnum-1].a
                    else:
                        prevOut = self.X[i]
                    
                    self.Layers[lnum].WGrad += np.dot(self.Layers[lnum].delta,prevOut.T)
                    self.Layers[lnum].bGrad += self.Layers[lnum].delta
                    lnum = lnum - 1
                
                if (gradDescType == GradDescType.MINIBATCH):
                    if (i % batchSize == 0):
                        self.UpdateGradsBiases(learningRate,lambda1, batchSize)

                if (gradDescType == GradDescType.STOCHASTIC):
                        self.UpdateGradsBiases(learningRate,lambda1, 1)

            if (gradDescType == GradDescType.BATCH):
                self.UpdateGradsBiases(learningRate,lambda1, self.X.shape[0])
            
            print("Iter = " + str(j) + " Loss = "+ str(loss))

    def UpdateGradsBiases(self, learningRate, lambda1, batchSize):
        # update weights and biases for all layers
        for ln in range(len(self.num_layers)):
            self.Layers[ln].W = self.Layers[ln].W - learningRate * (1/batchSize) * self.Layers[ln].WGrad 
            - learningRate * lambda1 * self.Layers[ln].W 
            self.Layers[ln].b = self.Layers[ln].b - learningRate * (1/batchSize) * self.Layers[ln].bGrad
            self.Layers[ln].clear_WBgradients()
