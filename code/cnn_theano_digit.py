import numpy as np
import theano
import theano.tensor as T
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from util import y2indicator,relu,get_digit_data,rearrange,init_filter,convpool,error_rate,get_digit_data_test

#getting train data from mnist handwritten digits dataset 
XTrain,YTrain = get_digit_data('../data/train.csv')
K = len(set(YTrain))
#coverting to an indicator matrix i.e. one hot encoding
YTrain_ind = y2indicator(YTrain)

#reading the validation set
XTest = get_digit_data_test('../data/test.csv')

#number of samples in the training set (~28000)
N = XTrain.shape[0]
#number of iterations for batch gradient descent
max_iter = 400
print_period = 100
#learning rate
lr = np.float32(0.0001)

#batch size for batch gradient descent
batch_sz = 500
#number of batches
n_batch = np.int32(N/batch_sz)
#hidden layer
M = 500
#pool size for max pooling
poolsz = (2,2)

#size of the first convolution filter with 20 output features, 1 input feature(black and white image) and size of 5X5
W1_shape = (20,1,5,5)
#size of the first convolution filter with 50 output features, 20 input feature(from previous step) and size of 5X5
W2_shape = (50,20,5,5)

W1_init,b1_init = init_filter(W1_shape,poolsz)
W2_init,b2_init = init_filter(W2_shape,poolsz)

#converting from convpool to ANN. Size of image is 28X28 and with filter of size 5X5, poolsize 2X2 we are left with 50X4X4 values for each image
W3_init = np.random.randn(W2_shape[0]*4*4,M)/np.sqrt(W2_shape[0]*4*4+M)
b3_init = np.zeros(M,dtype = np.float32)

#weight for hidden to output layer
W4_init = np.random.randn(M,K)/np.sqrt(M+K)
b4_init = np.zeros(K,dtype = np.float32)

#X is a 4 dimension matrix  of size Nx1X28X28
X = T.tensor4('X',dtype = 'float32')
#Y is the output prediction matrix of size NXK
Y = T.matrix('Y')
W1 = theano.shared(W1_init.astype(np.float32),'W1')
W2 = theano.shared(W2_init.astype(np.float32),'W2')
W3 = theano.shared(W3_init.astype(np.float32),'W3')
W4 = theano.shared(W4_init.astype(np.float32),'W4')
b1 = theano.shared(b1_init.astype(np.float32),'b1')
b2 = theano.shared(b2_init.astype(np.float32),'b2')
b3 = theano.shared(b3_init.astype(np.float32),'b3')
b4 = theano.shared(b4_init.astype(np.float32),'b4')

#Performing prediction on  2 convpool and one layered ANN
Z1 = convpool(X,W1,b1)
Z2 = convpool(Z1,W2,b2)
Z3 = relu(Z2.flatten(ndim=2).dot(W3)+b3)
pY = T.nnet.softmax(Z3.dot(W4)+b4)

#cost as cross entropy function
cost = (Y*T.log(pY)).sum()
#predicition as an argmax to convert it into a vector of size N by taking largest on each row
prediction = T.argmax(pY,1)

#updting theano variables at each step
update_W1 = W1 + lr*T.grad(cost,W1)
update_W2 = W2 + lr*T.grad(cost,W2)
update_W3 = W3 + lr*T.grad(cost,W3)
update_W4 = W4 + lr*T.grad(cost,W4)
update_b1 = b1 + lr*T.grad(cost,b1)
update_b2 = b2 + lr*T.grad(cost,b2)
update_b3 = b3 + lr*T.grad(cost,b3)
update_b4 = b4 + lr*T.grad(cost,b4)

#theano function to train taking XTrain and YTrain as inputs and updates to update the weeight while training
train = theano.function(inputs = [X,Y],updates = [(W1,update_W1),(W2,update_W2),(W3,update_W3),(W4,update_W4)])
#theano function to predict by taking X,Y(test set) as inputs and output as cost and prediction
get_prediction = theano.function(inputs = [X,Y],outputs = [cost,prediction])
#theano function for prediction on validation set taking input as X and output as the prediction
get_only_prediction = theano.function(inputs = [X],outputs = prediction)

for i in range(max_iter):
    for j in range(n_batch):
        #batch gradient descent
        XBatch = XTrain[j*batch_sz:j*batch_sz+batch_sz]
        YBatch = YTrain_ind[j*batch_sz:j*batch_sz+batch_sz]
#        print("Y Shape: "+str(YBatch.shape[0])+" "+str(YBatch.shape[1]))
#        print("X Shape: "+str(XBatch.shape))
        train(XBatch,YBatch)
        if(j%print_period == 0):
            cost,output_val = get_prediction(XBatch,YBatch)
            error_val = error_rate(np.argmax(YBatch,axis=1),output_val)
            print("Error for i="+str(i)+" and j="+str(j)+" and cost = "+str(cost)+" is ="+str(error_val))


#fetching the predition for validation set            
final_pred =  get_only_prediction(XTest)
#size of Test Set
NTest = XTest.shape[0]
f = open("Result.csv","w")

#writing the prediction for each sample in validation set in a file as ImageId,Label(Label is thedigit predicted from 0-9)
f.write("ImageId,Label\n")
for n in range(NTest):
    f.write(str(n+1)+","+str(final_pred[n])+"\n")
f.close()
