from keras.layers import Concatenate, Conv2D, MaxPooling2D

def inception(X, filters, stage, block, seed=None):
	"""
	The inception module will use at once:
	* 1x1conv (first component),
	* 3x3 conv (second component),
	* 5x5 conv (3rd component) and
	* maxpooling (3x3) followed by 1x1 conv (4th component)
	The 1x1 convs in 2nd to 4th components are beneficial for computational costs.


	Arguments:
	X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev) 
    (m = number of samples, n_H = height of the image, n_W = width of the image, n_C = number of channels, prev = previous layer)
    filters: python list of 4 integers, defining the number of filters for each CONV components 
    filters=[F1, F2, F3, F4] 
    stage: integer, used to name the layers, depending on their position in the network
    block: string/character, used to name the layers, depending on their position in the network
    seed: none by default, but if you want reproducible weight initialization you set it up with an integer. 
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)


	"""

	# defining name basis (to save the different entity in the network)
    conv_name_base = 'incep' + str(stage) + block + '_branch'
    mp_name_base = 'maxp' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3, F4 = filters

    # Save the input value
    # X_input = X

    # First component: 1x1 Conv
    X_1 = Conv2D(filters = F1, kernel_size = (1,1), strides=(1,1), padding='same', \
    	name=conv_name_base + '1a', \
    	kernel_initializer = glorot_uniform(seed=seed), \
    	activation='relu')(X)

    # Second component: 
    X_2 = Conv2D(filters=F2, kernel_size=(1,1), strides=(1,1), padding='same', \
    	name=conv_name_base + '2a', \
    	kernel_initializer= glorot_uniform(seed=seed), \
    	)(X)
    X_2 = Conv2D(filters=F2, kernel_size=(3,3), strides=(1,1), padding='same', \
    	name=conv_name_base + '2b', \
    	kernel_initializer= glorot_uniform(seed=seed), \
    	activation='relu')(X_2)

    # Third component:
    X_3 = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='same', \
    	name=conv_name_base + '3a', \
    	kernel_initializer= glorot_uniform(seed=seed), \
    	)(X)
    X_3 = Conv2D(filters=F3, kernel_size=(5,5), strides=(1,1), padding='same', \
    	name=conv_name_base + '3b', \
    	kernel_initializer= glorot_uniform(seed=seed), \
    	activation='relu')(X_3)

    # Fourth component:
    X_4 = MaxPooling2D(kernel_size=(3,3), strides=(1,1), padding='same', \
    	name=conv_name_base + '4a')(X)
    X_4 = Conv2D(filters=F4, kernel_size=(1,1), strides=(1,1), padding='same', \
    	name=conv_name_base + '4b', \
    	kernel_initializer= glorot_uniform(seed=seed), \
    	activation='relu')(X_4)

    # Concatenate the four components:
    X = Concatenate([X_1, X_2, X_3, X_4], axis=3) # concatenate over the channels (axis3)


