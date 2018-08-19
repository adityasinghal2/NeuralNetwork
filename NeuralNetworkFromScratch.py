import numpy as np
 
epochs = 10000    # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1
 
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [1]])
 
def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid
                                                # weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
bh=np.random.uniform(size=(1,hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize))
bout=np.random.uniform(size=(1,outputLayerSize))
 
for i in range(epochs):
    hidden_layer_input1=np.dot(X,Wh)
    hidden_layer_input=hidden_layer_input1 + bh
    H = sigmoid(hidden_layer_input)    # hidden layer results
    output_layer_input1=np.dot(H,Wz)
    output_layer_input= output_layer_input1+ bout
    Z = sigmoid(output_layer_input)    # output layer results

    E = (Y - Z)                                  # how much we missed (error)
    dZ = E * sigmoid_(Z)                            # delta Z
 
    dH = dZ.dot(Wz.T) * sigmoid_(H)             # delta H
    Wz +=  H.T.dot(dZ)                          # update output layer weights
    bout += np.sum(dZ, axis=0,keepdims=True)  # update output layer biases 
    Wh +=  X.T.dot(dH)                          # update hidden layer weights
    bh += np.sum(dH, axis=0,keepdims=True)# update hidden layer biases
    # print (Wh) 
print(Z) 