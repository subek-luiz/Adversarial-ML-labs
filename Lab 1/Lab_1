import sys

def main():
    # y = 2x + 0.3
    # x0 corresponds to x coordinate, x1 corresponds to y coordinate of a point
    # If the given point is below the y = 2x + 0.3 line, the neural network is 
    # to output a 0, if the point is above the line, it's output is to be 1.

    #------create some training data---------
    x0 = [1,2,3,4,5,6,7,8,9,10]
    x1 = [2.2,4.5,5.6,8.6,10.15,12.44,14.23,16.2,18.4,20.4]
    y = [0,1,0,1,0,1,0,0,1,1]  # expected outputs of the network (do not confuse it with
                               # y coordinate of a point

    #-------initialize weights and biases
    w0 = 0.1
    w1 = -0.23
    w2 = -0.2
    w3 = 0.7
    w4 = -0.1
    w5 = 0.5
    b0 = 0.22
    b1 = -0.1
    b2 = 0.3

    #-------- train the single neuron network
    # need 10000 epochs, with 0.001 learning rate
    for i in range(0,10000):
        loss = 0
        #Go through each sample and update the weights
        for j in range(0,len(y)):
            s0 = x0[j] * w0 + x1[j] * w1 + b0  # forward pass
            s1 = x0[j] * w2 + x1[j] * w3 + b1
            a0 = s0
            a1 = s1
            s2 = a0 * w4 + a1 * w5 + b2
            a2 = s2

            loss += 0.5 * (y[j] - a2)**2    # compute loss
            #TODO: Compute the gradients
            dw4 = -(y[j]-a2) * a0 #TODO
            dw5 = -(y[j]-a2) * a1 #TODO
            db2 = -(y[j]-a2) #TODO
            dw0 = -(y[j]-a2) * w4 * x0[j] #TODO
            dw1 = -(y[j]-a2) * w4 * x1[j] #TODO
            dw2 = -(y[j]-a2) * w5 * x0[j] #TODO
            dw3 = -(y[j]-a2) * w5 * x1[j] #TODO
            db0 = -(y[j]-a2) * w4 #TODO
            db1 = -(y[j]-a2) * w5 #TODO
            w0 = w0 - 0.001 * dw0  # update weights, biases
            w1 = w1 - 0.001 * dw1
            w2 = w2 - 0.001 * dw2  
            w3 = w3 - 0.001 * dw3
            w4 = w4 - 0.001 * dw4  
            w5 = w5 - 0.001 * dw5
            b0 = b0 - 0.001 * db0
            b1 = b1 - 0.001 * db1
            b2 = b2 - 0.001 * db2
        print('loss =',loss)

    # -----test for unknown data, on the trained network----------
    x0 = 2.7  # x coord. of point
    x1 = 6.0  # y coord. of point
    s0 = x0 * w0 + x1 * w1 + b0
    s1 = x0 * w2 + x1 * w3 + b1
    a0 = s0
    a1 = s1
    s2 = a0 * w4 + a1 * w5 + b2
    a2 = s2
    print('output for (',x0,',',x1,')= ',a2)

    x0 = 5.3  # x coord. of point
    x1 = 10.4  # y coord. of point
    s0 = x0 * w0 + x1 * w1 + b0
    s1 = x0 * w2 + x1 * w3 + b1
    a0 = s0
    a1 = s1
    s2 = a0 * w4 + a1 * w5 + b2
    a2 = s2
    print('output for (',x0,',',x1,')= ',a2)

if __name__ == "__main__":
    sys.exit(int(main() or 0))

