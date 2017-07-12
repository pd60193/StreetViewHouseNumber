# StreetViewHouseNumber
The aim of this project is to predict digits of house numbers captured by the Street View House Number(SVNH) dataset in the form of a 32X32 image.

## Dataset Link

Download the [train](http://ufldl.stanford.edu/housenumbers/train_32x32.mat) and [test](http://ufldl.stanford.edu/housenumbers/test_32x32.mat) data from the links. The **data** folder already has the test data. However the train data is too large and you should download it before running the scripts. TLoading the MAT file creates two varaibles -:

- **X** which is a 4-D matrix containing the images
- **y** is a vector of class labels

To access the images, X(:,:,:,i) gives the i-th 32-by-32 RGB image, with class label y(i)

##Code

The file **code/cnn_theano.py** contains the code for training on the SVNH dataset. The code has been written in Python using theano. It performs a batch gradient descent on a convolutional neural network propgating the error backwards at each layer.

- We use two Convolution-Pooling layers with pool size as 2X2 and filter sizes as-:

    * 20 X 3 X 5 X 5 -: This signifies that first filter will take in input an RGB image(R,G,B being the three input features) and       outputs an image with 20 features. The filter size is 5X5.
    A 32X32 image with each pixel having 3 values (R,G,B) is converted to 28X28 image with 20 values(features). The size of image reduces as we do not add extra pixels on the image before performing convolution. Following this we perform a max-pooling. Max Pooling converts the image from 28X28X20 to 14X14X20.

    * 50 X 20 X 5 X 5 -: This signifies the size of second filter which takes input an image with 20 features and outputs an image with 50 features. The size of this filter is 5X5 as well.
    A 14X14 image with each pixel having 20 values(features) is converted to 10X10 image with 50 values(features). The size of image reduces as we do not add extra pixels on the image before performing convolution. Following this we perform a max-pooling. Max Pooling converts the image from 10X10X50 to 5X5X50.

- After perfoming convolution and pooling we decode an image for a Neural Network with 5X5X50 i.e. 1250 input features. Hence each image is represented by 1250 values. 

