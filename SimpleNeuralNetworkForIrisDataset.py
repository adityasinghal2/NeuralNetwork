import numpy as np
import json

import pandas as pd
import matplotlib.pyplot as plt

class Network():

    def __init__(self):
        np.random.seed(1)
        self.epochs = 600    # Number of iterations
        self.inputLayerSize, self.hiddenLayerSize, self.outputLayerSize = 4, 5, 3
        # print(y[0])
        self.Wh = np.random.uniform(size=(self.inputLayerSize, self.hiddenLayerSize))
        self.bh=np.random.uniform(size=(1,self.hiddenLayerSize))
        self.Wz = np.random.uniform(size=(self.hiddenLayerSize,self.outputLayerSize))
        self.bout=np.random.uniform(size=(1,self.outputLayerSize))
    def sigmoid (self,x): return 1/(1 + np.exp(-x))      # activation function

    def sigmoid_(self,x): return x * (1 - x)             # derivative of sigmoid
    def save_model(self,model_dic):
        with open('model.json','w') as f:
            f.write(json.dumps(model_dic))
    def load_data(self):
        #Set working directory and load data
        iris = pd.read_csv('irisdataset.csv')
        iris = iris.sample(frac=1).reset_index(drop=True)
        print (iris)
        #Create numeric classes for species (0,1,2) 
        # print (iris)
        iris.loc[iris['species']=='virginica','species']=2
        iris.loc[iris['species']=='versicolor','species']=1
        iris.loc[iris['species']=='setosa','species'] = 0
        # print (pd.Series.iris.values.tolist())

        labels = iris["species"].tolist()
        iris.drop(columns=["species"],inplace=True)
        temp = []
        for row in iris.iterrows():
            index, data = row
            temp.append(data.tolist())
        # iris = iris(columns = ['sepal_length','sepal_width','petal_length','petal_width']).tolist()[1:]
        # print (temp)
        # print (labels)

        y = []
        for i in labels:
            if i==0:
                y.append([1,0,0])
            elif i==1:
                y.append([0,1,0])
            else:
                y.append([0,0,1])
        print (temp,y)
        return temp,y

    def forwordPropagation(self,X):
        hidden_layer_input1=np.dot(X,self.Wh)
        hidden_layer_input=hidden_layer_input1 + self.bh
        H = self.sigmoid(hidden_layer_input)    # hidden layer results
        output_layer_input1=np.dot(H,self.Wz)
        output_layer_input= output_layer_input1+ self.bout
        Z = self.sigmoid(output_layer_input)    # output layer results
        return H,Z

    def Backpropagation(self,H,Z,dZ,X):
        dH = dZ.dot(self.Wz.T) * self.sigmoid_(H)             # delta H
        self.Wz +=  H.T.dot(dZ)                          # update output layer weights
        self.bout += np.sum(dZ, axis=0,keepdims=True)  # update output layer biases 
        self.Wh +=  X.T.dot(dH)                          # update hidden layer weights
        self.bh += np.sum(dH, axis=0,keepdims=True)# update hidden layer biases
    def predict(self,X):
        return self.forwordPropagation(X)

def main():
    n = Network()
    temp,y = n.load_data()
    for i in range(n.epochs):
        for i in range(0,len(temp),2):
            X = np.array([temp[i]])
            Y = np.array(y[i])
            H,Z = n.forwordPropagation(X)
            E = (Y - Z)                                  # how much we missed (error)
            dZ = E * n.sigmoid_(Z)                            # delta Z
            n.Backpropagation(H,Z,dZ,X)

            # print (Wh) 
    print(Z)
    print(type(n.Wh.tolist()))
    model_dic = {
        'wh':n.Wh.tolist(),
        'wo':n.Wz.tolist(),
        'bh':n.bh.tolist(),
        'bo':n.bout.tolist()
    }
    n.save_model(model_dic)
    # print(model_dic)
    print (n.predict([[6.5,3.2,5.1,2.0]]))

if __name__== "__main__":
    main()
