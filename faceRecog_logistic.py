
import os
import cv2
import numpy as np
import h5py
def  sigmoid(z):
	ans = np.zeros((z.shape[0],1))
	#print(z,"Z")
	ans = 1 / (1+ np.exp(-z))
	#print(ans,"ANS")
	return ans
def find_cost(X, Y,W,pred):
	#print(pred.shape)
	pred = np.zeros((Y.shape[0],1))
	z = np.zeros((Y.shape[0],1))
	z = np.dot(X.T,W)
	z = z.reshape(Y.shape[0],1)
	#print(z.shape,"Z")
	m = 0
	m = Y.shape[0]
	cost = 0
	grad = np.zeros((X.shape[0],1))
	#print(sigmoid(1-z),"sigma")
	cost = -1/m * ( np.dot(np.log(sigmoid(z)).T,Y) + np.dot(np.log(sigmoid(1-z).T),(1-Y)))
	#cost = cost[0]
	#print(cost,"COST")
	grad = np.dot( X , sigmoid(z) - Y ) / m
	#print(grad.shape,"GRAD1")
	return cost,grad
def train(X,Y,k,num_itr,W,learning_rate):
	cost=0
	#grad = np.zeros((X.shape[0],1))
	c = np.zeros((Y.shape[0],1))
	w = np.zeros((W.shape[0],1))
	#pred = np.zeros((X.shape[0],1))
	#print(c.shape,X.shape,Y.shape,W.shape)
	for i in range(k):
		for j in range(num_itr):
			c = (Y == i)
			w = W[:,i]
			pred = np.argmax(np.dot(X.T,W),axis = 1)
			pred = pred.reshape(Y.shape[0],1)
			#print(pred.shape,pred,"JHBJCHB")
			cost,grad = find_cost(X,c,w,pred)
			#W[:,i] = W[:,i].reshape(X.shape[0],1)
			#print(grad)
			grad =grad[:,0]
			#print(grad)
			#print(grad.shape,"WWW")
			#print(cost)

			#W = W.T
			W[:,i] =W[:,i] -  learning_rate*grad
	#		print(W)
	h5f = h5py.File('weights.h5', 'w')
	h5f.create_dataset('W', data=W)
	h5f.close()
	return W

def test_accuracy(W):
	directory = "test"
	faces , faceID = labels_for_training_data(directory)
	x = np.array(faces)
	y = np.array(faceID)
	Y = y.reshape(len(faceID),1)
	X = x.reshape(len(faces),64,64,3)
	X = X.reshape(X.shape[0],-1).T
	#print(X.shape[0])
	#print(Y.shape)
	X = X / 255.0
	z = np.dot(W.T,X)
	prediction = np.argmax(sigmoid(z),axis = 0 )
	#print(prediction.shape , Y.shape[0])
	prediction = prediction.reshape(Y.shape[0],1)
	#print((prediction==Y)*1)
	check = np.sum((prediction == Y)*1)
	#print(check,len(faces))
	accuracy = float(check) / len(faces) * 1.0
	return accuracy

def predict(imagefile):
	h5f = h5py.File('weights.h5', 'r')
	W = h5f['W'][:]
	h5f.close()
	img = cv2.imread(imagefile)
	nimg = cv2.resize(64,64,3)
	X = nimg.reshape(12288,1)
	z = np.dot(W.T,X)
	prediction = np.argmax(sigmoid(z),axis = 1 )
	print(prediction)


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
if __name__ == '__main__':
	print("1. Train neural net\n2. Predict")
	response = int(input("Enter your response: "))
	if response  == 1 :
		directory = "Train_data"
		faces,faceID = labels_for_training_data(directory)
		x = np.array(faces)
		y = np.array(faceID)
		Y = y.reshape(len(faceID),1)
		X = x.reshape(len(faces),64,64,3)
	#	print(X.shape)
		X = X.reshape(X.shape[0],-1).T
	#	print(X.shape[0])
	#	print(Y.shape)
		X = X / 255.0

		W = np.zeros((X.shape[0],3))
		print("Training.....")
		parameters = train(X,Y,3,1000,W,0.1)

		accuracy = test_accuracy(parameters)
		#print(parameters)
		print("The accuracy of the trained model is %.2f%s" % (100*accuracy, "%"))

		#hf = h5py.File('data.h5','w')
		#hf.create_dataset('data',data = param)
		#<HDF5 dataset "data" : shape(12288,3) , type " <f8" >
		#hf.close()


#hf = h5py.File('data.h5','r')
#print(hf.keys())
	elif response == 2 :
		imagefile = input("Enter the image to make prediction upon : ")
		predict(imagefile)

		