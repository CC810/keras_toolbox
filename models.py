from residual_blocks import *
from keras.initializers import he_normal
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, AveragePooling2D, Dense
from keras.models import Model


def ResNet50(input_shape = (64, 64, 3), classes):
    """
    Implementation of the popular ResNet50 the following architecture with residual blocks of 3 layers:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block_3(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block_3(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block_3(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 
    X = convolutional_block_3(X, f = 3, filters = [128,128,512], stage = 3, block = 'a')
    X = identity_block_3(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block_3(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block_3(X, 3, [128,128,512], stage=3, block='d')

    # Stage 4 
    X = convolutional_block_3(X, f = 3, filters = [256, 256, 1024], stage = 4, block ='a')
    X = identity_block_3(X, 3, [256, 256, 1024], stage = 4, block='b')
    X = identity_block_3(X, 3, [256, 256, 1024], stage = 4, block='c')
    X = identity_block_3(X, 3, [256, 256, 1024], stage = 4, block='d')
    X = identity_block_3(X, 3, [256, 256, 1024], stage = 4, block='e')
    X = identity_block_3(X, 3, [256, 256, 1024], stage = 4, block='f')

    # Stage 5 
    X = convolutional_block_3(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'a')
    X = identity_block_3(X, 3, [512, 512, 2048], stage = 5, block = 'b')
    X = identity_block_3(X, 3, [512, 512, 2048], stage = 5, block = 'c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), name = 'avg_pool')(X)
   
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = he_normal(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name= 'ResNet50')

    return model