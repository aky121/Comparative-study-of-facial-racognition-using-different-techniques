import numpy as np
import os
import cv2
import h5py

def labels_for_training_data(directory):
        faces=[]
        faceID=[]
    
        for path,subdirnames,filenames in os.walk(directory):
            for filename in filenames:
                if filename.startswith("."):
                    print("Skipping system file")#Skipping files that startwith .
                    continue
    
                id=os.path.basename(path)#fetching subdirectory names
                img_path=os.path.join(path,filename)#fetching image path
                print("img_path:",img_path)
                print("id:",id)
                test_img=cv2.resize(cv2.imread(img_path),(64,64))#loading each image one by one
                if test_img is None:
                    print("Image not loaded properly")
                    continue
               
                faces.append(test_img)
                faceID.append(int(id))
        return faces,faceID

def initialize_parameters(num_people):
    W1 = np.random.randn(2000,12288)*0.01
    b1 = np.zeros((2000,1))
    W2 = np.random.randn(2000,2000)*0.01
    b2 = np.zeros((2000,1))
    W3 = np.random.randn(num_people , 2000)*0.01
    b3 = np.zeros((num_people,1))
    parameters = {
        "W1" : W1,
        "b1" : b1 ,
        "W2" : W2,
        "b2" : b2,
        "W3" : W3,
        "b3" : b3
    }
    return parameters

def sigmoid(z):
    ans = 1.0 / (1.0 + np.exp(-z));
    return ans

def forward_prop(X,parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)
    cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"Z3":Z3,"A2":A2,"A3":A3}
    return cache

def compute_cost(A3,Y):
    m = A3.shape[1]
    #print(np.log(A3))
    #print(np.log(1-A3))
    cost = -1/m * (   np.dot(Y,np.log(A3).T) + np.dot((1-Y) , np.log(1-A3).T)  )
    return cost

def back_prop(parameters,cache,X,y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    print(A1)
    dZ3 = A3 - y
    dW3 = 1/m*(np.dot(dZ3, A2.T))
    db3 = 1/m*(np.sum(dZ3,axis = 1,  keepdims = True))
    dZ2 = np.dot(W3.T, dZ3) * (1-np.power(A2, 2))

    dW2 = 1/m*(np.dot(dZ2,A1.T))
    db2 = 1/m *(np.sum(dZ2,axis = 1,  keepdims = True))
    dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1, 2))

    dW1 = 1/m*(np.dot(dZ1,X.T))
    db1 = 1/m *(np.sum(dZ1,axis = 1,  keepdims = True))
    
    grads = {"dW1": dW1,"dW2": dW2,"dW3": dW3,"db1":db1,"db2":db2,"db3":db3}
    return grads

def update_parameters(parameters,grads,learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]


    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    parameters = {"W1":W1,"W2":W2,"W3":W3,"b1":b1,"b2":b2,"b3":b3}
    return parameters

def train(X,Y,num_layers,num_neurons,num_people):
        
    parameters = initialize_parameters(num_people)
    y = np.zeros((num_people,X.shape[1]))
    #print(y.shape)
    #print(Y)
    #print(y[2][50])
    for i in range(Y.shape[0]):
        j = Y[i]
        #print(j)
        y[j,i] = 1

    for i in range(100 ):
        print("In Big letters epoch number: ", i)
        cache = forward_prop(X,parameters)
        cost = compute_cost(cache["A3"],y)
        grads = back_prop(parameters,cache,X,y)
        parameters = update_parameters(parameters,grads,learning_rate = 1.5)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    h5f = h5py.File('weights.h5', 'w')
    h5f.create_dataset('W1', data=W1)
    h5f.create_dataset('b1', data=b1)
    h5f.create_dataset('W2', data=W2)
    h5f.create_dataset('b2', data=b2)
    h5f.create_dataset('W3', data=W3)
    h5f.create_dataset('b3', data=b3)
    h5f.close()
    
    return parameters
def test_accuracy(parameters):

        directory = "test"
        people , labels = labels_for_training_data(directory)
        people = np.array ( people)
        X  = people . reshape(len(people),64,64,3)
        X = X.reshape(X.shape[0],-1).T
        X = X/ 255
        Y = np.array(labels)
        Y = Y.reshape(len(labels),1)

        cache = forward_prop(X, parameters)
        out = cache["A3"]
        index = np.argmax(out, axis=1)
        y = index.T
        check = np.sum((Y==y)*1)
        accuracy = check/len(Y)
        return accuracy

        
def predict(imagepath):

    h5f = h5py.File('weights.h5','r')
    W1 = h5f['W1'][:]
    b1 = h5f['b1'][:]
    W2 = h5f['W2'][:]
    b2 = h5f['b2'][:]
    W3 = h5f['W3'][:]
    b3 = h5f['b3'][:]
    parameters = {"W1":W1,"W2":W2,"W3":W3,"b1":b1,"b2":b2,"b3":b3}
    h5f.close()

    test_img=cv2.resize(cv2.imread(imagepath),(64,64))
    A0 = test_img.reshape(12288, 1)
    cache = forward_prop(A0, parameters)
    out = cache["A3"]
    index = np.argmax(out)
    y = np.zeros((out.shape[0],1))
    y[index] = 1
    print("The Face is recognisised as: ", index)

if __name__ == '__main__':
    print("1. Train neural net\n2. Predict")
    response = int(input("Enter option number: "))

    if response == 1:
            directory = "Train_data"
            people , labels = labels_for_training_data(directory)
            num_people = len(os.listdir(directory))
            #print(len(people))
            people = np.array ( people)
            X  = people . reshape(len(people),64,64,3)
            X = X.reshape(X.shape[0],-1).T
            X = X/ 255
            #print(X.shape)
            Y = np.array(labels)
            Y = Y.reshape(len(labels),1)
            #print(Y.shape)
            num_layers = 3
            num_neurons = 20000
            parameters = train(X , Y , num_layers,num_neurons,num_people )
            print(parameters)
            accuracy = test_accuracy(parameters)
            print("The accuracy of the trained model is %.2f%s" % (100*accuracy, "%"))
            
    elif response == 2:
            imagefile = input("Enter the image file to predict: ")
            predict(imagefile)

    else:
            print("Invalid entry")

