import sys
from Network import Network
from GradDescType import *
from ActivationType import *
from Utils import Utils
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(pred_func, X, y): 
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    xdata = np.c_[xx.ravel(), yy.ravel()]
    xdatanp = xdata.reshape(xdata.shape[0],xdata.shape[1],1)
    #print(xdatanp.shape)
    Z = [pred_func(xdatanp[x]) for x in range(0,len(xdatanp))] 
    Z = np.array(Z)
    exp_scores = np.exp(Z)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    Z = np.argmax(probs, axis=1)
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 
    plt.show()

def main():
    utils = Utils()
    X, Y = utils.init_data()  # initialize data
    #X = utils.normalizeData(X)
    trainX = X.reshape((X.shape[0],X.shape[1],1)) 
    trainY = np.zeros((len(Y),2))
    for i in range(0,len(Y)):
        if Y[i] == 1:
            trainY[i,0] = 1
            trainY[i,1] = 0
        else:
            trainY[i,0] = 0
            trainY[i,1]= 1
    trainY = trainY.reshape((X.shape[0],X.shape[1],1)) 
    numLayers = [30,2]         
    NN = Network(trainX,trainY,numLayers,1.0,ActivationType.TANH, ActivationType.SOFTMAX) # try different activation functions
    NN.Train(5000,0.10,0.0, GradDescType.MINIBATCH,1)

    #------------ compute accuracy----------
    accuracy = 0
    for i in range(len(trainX)):
        pred = NN.forward(trainX[i])
        if (pred.argmax() == 0 and trainY[i,0,0] == 1) or \
            (pred.argmax() == 1 and trainY[i,0,0] == 0):
            accuracy = accuracy + 1 
    accuracy_percent = accuracy/len(trainX)
    print('accuracy =', accuracy_percent)
    plot_decision_boundary(lambda x: NN.forward(x), X, Y)
    

if __name__ == "__main__":
    sys.exit(int(main() or 0))
