import numpy as np
from scipy.io import loadmat
from theano.tensor.nnet import conv2d
import theano.tensor.signal.pool as pool
import theano.tensor as T
import pandas as pd

def error_rate(T,P):
    return np.mean(T!=P)

def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    T = np.zeros((N,K),dtype=np.float32)
    for n in range(N):
        T[n,Y[n]]=1
    return T

def relu(X):
    return X*(X>0)

def loadData(path):
    data = loadmat(path)
    X = data['X']
    Y = (data['y']-1).flatten()
    return X[:,:,:,:20000],Y[:20000]

def rearrange(X):
    N = X.shape[-1]
    out = np.zeros((N,X.shape[2],X.shape[1],X.shape[0]),dtype = np.float32)
    for i in range(N):
        for j in range(3):
            out[i,j,:,:] = X[:,:,j,i]
    return (out/255).astype(np.float32)

def init_filter(shape,poolsz):
    W = np.random.randn(*shape)/np.sqrt(np.prod(shape[1:])+shape[0]*np.prod(shape[2:])/np.prod(poolsz))
    b = np.zeros(shape[0],dtype = np.float32)
    return W.astype(np.float32),b

def convpool(X,W,b,poolsz=(2,2)):
    conv_out = conv2d(input = X, filters = W)
    pooled_out = pool.pool_2d(conv_out,ws = poolsz,ignore_border=True)
    return relu(pooled_out+b.dimshuffle('x',0,'x','x'))
def get_digit_data(path):
    data = pd.read_csv(path).as_matrix().astype(np.float32)
    out = np.zeros((data.shape[0],1,28,28),dtype = np.float32)
    Y = data[:,0]
    X = data[:,1:]
    for eachitem in range(data.shape[0]):
        for j in range(28):
            out[eachitem,0,j,:]=X[eachitem,j*28:j*28+28]
    return (out/255).astype(np.float32),Y.astype(np.int32)        
    
def get_digit_data_test(path):
    data = pd.read_csv(path).as_matrix().astype(np.float32)
    out = np.zeros((data.shape[0],1,28,28),dtype = np.float32)
    X = data
    for eachitem in range(data.shape[0]):
        for j in range(28):
            out[eachitem,0,j,:]=X[eachitem,j*28:j*28+28]
    return (out/255).astype(np.float32)

