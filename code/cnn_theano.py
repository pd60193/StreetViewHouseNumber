import numpy as np
import theano
import theano.tensor as T
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from util import y2indicator,relu,loadData,rearrange,init_filter,convpool,error_rate


#loading the train dataset
#train data is no checked in. Get from http://ufldl.stanford.edu/housenumbers/train_32x32.mat and place in data folder
XTrain,YTrain = loadData('../data/train_32x32.mat')
K = len(set(YTrain))
XTrain = rearrange(XTrain)

#creating the indicator matrix i.e one hot encoding
YTrain_ind = y2indicator(YTrain)
#print(str(YTrain_ind.shape[0]))

#loading the test data
XTest,YTest = loadData('../data/test_32x32.mat')
XTest = rearrange(XTest)
YTest_ind = y2indicator(YTest)

#N -> number of image samples for training
N = XTrain.shape[0]
#max_iter -> number of iterations
max_iter = 500
print_period = 20

#lr -> learning rate
lr = np.float32(0.0001)

#batch size for batch gradient descent
batch_sz = 500
#number of batches
n_batch = np.int32(N/batch_sz)
#M -> number of hidden units
M = 500
#pool size for max pooling
poolsz = (2,2)

#W1 -> filter 1 with 20 output features, 3 input features(r,g,b), and a size of 5X5
W1_shape = (20,3,5,5)
#W1 -> filter 1 with 50 output features, 20 input features(from prev step), and a size of 5X5
W2_shape = (50,20,5,5)

#initializng the filters 
W1_init,b1_init = init_filter(W1_shape,poolsz)
W2_init,b2_init = init_filter(W2_shape,poolsz)

#decoding the image into a Neural Network after two convpool layers. A 32X32 image becomes 5X5 after two convpool filters of size 5X5 and image becoming smaller 
W3_init = np.random.randn(W2_shape[0]*W2_shape[2]*W2_shape[3],M)/np.sqrt(W2_shape[0]*W2_shape[2]*W2_shape[3]+M)
b3_init = np.zeros(M,dtype = np.float32)

#Weight for the hidden to output layer
W4_init = np.random.randn(M,K)/np.sqrt(M+K)
b4_init = np.zeros(K,dtype = np.float32)

#X is N X 3 X 32 X 32
X = T.tensor4('X',dtype = 'float32')
#Y is N X K where K is from 0 to 9
Y = T.matrix('Y')

#defining theano shared variables
W1 = theano.shared(W1_init.astype(np.float32),'W1')
W2 = theano.shared(W2_init.astype(np.float32),'W2')
W3 = theano.shared(W3_init.astype(np.float32),'W3')
W4 = theano.shared(W4_init.astype(np.float32),'W4')
b1 = theano.shared(b1_init.astype(np.float32),'b1')
b2 = theano.shared(b2_init.astype(np.float32),'b2')
b3 = theano.shared(b3_init.astype(np.float32),'b3')
b4 = theano.shared(b4_init.astype(np.float32),'b4')

#calculating two convpools and an ANN at the end to decode the output.
Z1 = convpool(X,W1,b1)
Z2 = convpool(Z1,W2,b2)
Z3 = relu(Z2.flatten(ndim=2).dot(W3)+b3)
pY = T.nnet.softmax(Z3.dot(W4)+b4)

#calculating cost
cost = (Y*T.log(pY)).sum()
#prediction argmax to convert indicator to vector of size N
prediction = T.argmax(pY,1)

#updating the theano variables for each step
update_W1 = W1 + lr*T.grad(cost,W1)
update_W2 = W2 + lr*T.grad(cost,W2)
update_W3 = W3 + lr*T.grad(cost,W3)
update_W4 = W4 + lr*T.grad(cost,W4)
update_b1 = b1 + lr*T.grad(cost,b1)
update_b2 = b2 + lr*T.grad(cost,b2)
update_b3 = b3 + lr*T.grad(cost,b3)
update_b4 = b4 + lr*T.grad(cost,b4)
#theano train function with input and updates
train = theano.function(inputs = [X,Y],updates = [(W1,update_W1),(W2,update_W2),(W3,update_W3),(W4,update_W4)])
#theano predicition with input as X,Y(for cost calculation on test data) ad outputs cost and predicition
get_prediction = theano.function(inputs = [X,Y],outputs = [cost,prediction])

for i in range(max_iter):
    for j in range(n_batch):
        #batch gradient descent
        XBatch = XTrain[j*batch_sz:j*batch_sz+batch_sz]
        YBatch = YTrain_ind[j*batch_sz:j*batch_sz+batch_sz]
        train(XBatch,YBatch)
        if(j%print_period == 0):
            cost,output_val = get_prediction(XTest,YTest_ind)
            error_val = error_rate(YTest,output_val)
        print("Error for i="+str(i)+" is ="+str(error_val))


            
