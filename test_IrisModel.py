import json
import numpy as np
def load_model():
    with open('model.json','r') as f:
        model = f.read()
        data = json.loads(model)
    return data
    print (type(data))

def sigmoid (x): return 1/(1 + np.exp(-x))

def forwordPropagation(Wh,Wz,bh,bout,X):
    hidden_layer_input1=np.dot(X,Wh)
    hidden_layer_input=hidden_layer_input1 + bh
    H = sigmoid(hidden_layer_input)    # hidden layer results
    output_layer_input1=np.dot(H,Wz)
    output_layer_input= output_layer_input1+ bout
    Z = sigmoid(output_layer_input)    # output layer results
    return Z
data = load_model()
# count = np.sum(np.argmax(predOutput,axis=1) == np.argmax(YTest,axis=1))


#print accuracy
# print ('Accuracy : ',(float(count)/float(YTest.shape[0])))
output = forwordPropagation(data['wh'],data['wo'],data['bh'],data['bo'],[[5.1,3.5,1.4,0.2],[6.9,3.1,4.9,1.5],[4.9,3.1,1.5,0.1],[5.8,2.8,5.1,2.4],[6.1,3.0,4.6,1.4],[4.9,2.5,4.5,1.7]])
print(output)
