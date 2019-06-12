
from keras import layers
from keras.layers import Add, Activation, BatchNormalization, Flatten, Conv2D
from keras.initializers import he_normal


def identity_block_3(X, f, filters, stage, block, seed=None):
    """
    Implementation of the identity block: 
    X, input from the previous layer will be use for two routes:
    1. the main one, which goes through 3 layers of Conv/BatchNormalization/Relu
    2. the short cut one, which will be added back into the main path (before last Relu activation) to reincorporate info from the initial X
    
    Arguments:
    X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev) 
    (m = number of samples, n_H = height of the image, n_W = width of the image, n_C = number of channels, prev = previous layer)
    f: integer, specifying the shape of the middle CONV's window for the main path
    filters: python list of 3 integers, defining the number of filters for each CONV layers of the main path
    stage: integer, used to name the layers, depending on their position in the network
    block: string/character, used to name the layers, depending on their position in the network
    seed: none by default, but if you want reproducible weight initialization you set it up with an integer. 
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis (to save the different entity in the network)
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', \
               kernel_initializer = he_normal(seed=seed))(X) # He initializer: better for relu activation
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', \
              kernel_initializer = he_normal(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1,1), strides= (1,1), padding = 'valid', name = conv_name_base + '2c', \
              kernel_initializer = he_normal(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut]) # NB: you cannot use + symbol
    X = Activation('relu')(X)
  
    
    return X



def identity_block_2(X, f, filters, stage, block, seed=None):
    """
    Same as the identity_block_3 but with 2 layers of Conv/BatchNormalization/Relu in the main path instead of 3

    Arguments:
    X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev) 
    (m = number of samples, n_H = height of the image, n_W = width of the image, n_C = number of channels, prev = previous layer)
    f: integer, specifying the shape of the middle CONV's window for the main path
    filters: python list of 3 integers, defining the number of filters for each CONV layers of the main path
    stage: integer, used to name the layers, depending on their position in the network
    block: string/character, used to name the layers, depending on their position in the network
    seed: none by default, but if you want reproducible weight initialization you set it up with an integer. 
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis (to save the different entity in the network)
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', \
               kernel_initializer = he_normal(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', \
              kernel_initializer = he_normal(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
  
    
    return X



def convolutional_block_3(X, f, filters, stage, block, s = 2, seed=None):
    """
    Implementation of the convolutional block:
    X, input from the previous layer will be use for two routes:
    1. the main one, which goes through 3 layers of Conv/BatchNormalization/Relu and leading to different dimension as X_input
    2. the short cut one, which after one pass through conv/batchnorm (to get the dimensions of X at the last layer of the main path)
    will be added back into the main path (before last Relu activation).
    
    Arguments:
    X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev) 
    (m = number of samples, n_H = height of the image, n_W = width of the image, n_C = number of channels, prev = previous layer)
    f: integer, specifying the shape of the middle CONV's window for the main path
    filters: python list of 3 integers, defining the number of filters for each CONV layers of the main path
    stage: integer, used to name the layers, depending on their position in the network
    block: string/character, used to name the layers, depending on their position in the network
    seed: none by default, but if you want reproducible weight initialization you set it up with an integer. 
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = he_normal(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', name= conv_name_base + '2b', \
             kernel_initializer = he_normal(seed=seed))(X) 
    X = BatchNormalization(axis=3, name= bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name= conv_name_base + '2c', \
               kernel_initializer = he_normal(seed=seed))(X) 
    X = BatchNormalization(axis=3, name= bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid',name=conv_name_base + '1', \
                       kernel_initializer = he_normal(seed=seed))((X_shortcut)) 
    X_shortcut = BatchNormalization(axis=3, name= bn_name_base + '1')((X_shortcut))

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X
