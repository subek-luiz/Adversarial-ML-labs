import numpy as np
from ActivationType import ActivationType

class Layer(object):  # represents one layer of neurons in an NN
    def __init__(self,num_neurons,num_neurons_prev_layer, last_layer= False,drop_out = 0.2,activation_type=ActivationType.SIGMOID):
        self.num_neurons = num_neurons
        self.last_layer = last_layer
        self.numNeurons_pre_Layer = num_neurons_prev_layer
        self.activation_function = activation_type 
        self.drop_out = drop_out
        self.delta = np.zeros((num_neurons,1))
        self.a = np.zeros((num_neurons,1)) # actual output from layer
        self.derivAF = np.zeros((num_neurons,1)) # derivative of Activation function
        self.W = np.random.randn(num_neurons, num_neurons_prev_layer)/ np.sqrt(num_neurons_prev_layer) # initialize weight matrix randomly
        self.b = np.zeros((num_neurons,1))
        self.WGrad = np.zeros((num_neurons,num_neurons_prev_layer))
        self.bGrad = np.zeros((num_neurons,1))  # gradient for biases

    def forward(self,input_data):
        sum = np.dot(self.W,input_data) + self.b
        aa = 0
        if (self.activation_function == ActivationType.NONE):
            self.a = sum
            self.derivAF = 1
        if (self.activation_function == ActivationType.SIGMOID):
            self.a = self.sigmoid(sum)
            self.derivAF = self.a * (1 - self.a)
        if (self.activation_function == ActivationType.TANH):
            self.a = self.TanH(sum)
            self.derivAF = (1 - self.a*self.a)
        if (self.activation_function == ActivationType.RELU):
            self.a = self.Relu(sum)
            self.derivAF = 1.0 * (self.a > 0)
        if (self.activation_function == ActivationType.SOFTMAX):
            self.a = self.Softmax(sum)
            self.derivAF = None  # we do delta computation in Softmax layer
        if (self.last_layer == False):
            zeroout = np.random.binomial(1,self.drop_out,(self.num_neurons,1))/self.drop_out
            # which neurons to zero out
            self.a = self.a * zeroout
            self.derivAF = self.derivAF * zeroout
        return self.a
    
    def Linear(self,x):
        return x  # output same as input
            
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))  # np.exp makes it operate on entire array
    
    def TanH(self, x):
        return np.tanh(x)

    def Relu(self, x):
        return np.maximum(0,x)

    def Softmax(self, x):
        ex = np.exp(x)
        return ex/ex.sum()

    def clear_WBgradients(self):  # zero out weight and bias gradients
        self.WGrad =  0 
        self.bGrad = 0 
