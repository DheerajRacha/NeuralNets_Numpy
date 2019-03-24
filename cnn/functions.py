import numpy as np

############ Activation Functions ##############
def sigmoid(x):
	return (1.0/(1.0+np.exp(-x)))

def ReLu(x):
	return np.maximum(x,0)

def softmax(x):
	q = np.exp(x-np.max(x))
	return q/q.sum()


############### Convolution Function ###################

def conv2d( input  ,  Kernel , stride , padding , non_linearialty ):
	kernel = Kernel[0]
	bias = Kernel[1]
	img_height , img_width = input.shape[0],input.shape[1]
	kernel_size = kernel.shape[0]

	if padding == -1:
		padding = (kernel_size-1)/2
	# print(img_height,kernel_size,kernel.shape)
	output = np.zeros(( int((img_height-kernel_size+2*padding)/stride) + 1,int((img_width-kernel_size+2*padding)/stride) + 1 ))
	#print(output.shape[0])

	#input = np.pad(input,[(padding,),(padding,),()],mode='constant',constant_values=0)

	for i in range( int((img_height-kernel_size+2*padding)/stride) + 1 ):
		for j in range(int((img_width-kernel_size+2*padding)/stride) + 1):
			current_patch = input[i*stride:i*stride +kernel_size,j*stride:j*stride + kernel_size]
			output[i][j] = np.sum(current_patch*kernel) + bias
			# print(output[i][j],i,j)

	if non_linearialty == 'ReLu':
		activation_map = ReLu(output)
	if non_linearialty == 'sigmoid':
		activation_map = sigmoid(output)
	if non_linearialty == 'softmax':
		activation_map = softmax(output)

	return np.array(output) , np.array(activation_map)


################### Pooling Function #########################

def pooling(input,type,pool_size):
	img_height , img_width = input.shape[0],input.shape[1]
	output = np.zeros(( int((img_height-pool_size )/pool_size)+1,int((img_width-pool_size)/pool_size)+1 ) )

	for i in range(int((img_height-pool_size)/pool_size) +1):
		for j in range(int((img_width -pool_size)/pool_size)+1):

			current_patch = input[i*pool_size:i*pool_size + pool_size,j*pool_size:j*pool_size + pool_size]
			
			if type == 'max':
				output[i][j] = np.max(current_patch)

			if type == 'avg':
				output[i][j] = np.mean(current_patch)

			if type  == 'min':
				output[i][j] = np.min(current_patch) 



	return output

################### Convolutional Layer ########################

def convlayer( input , Kernels , stride , padding , non_linearialty):
	kernels = Kernels[0]
	biases = Kernels[1]
	output_a = []
	output_z = []
	i = 1
	for kernel,bias in zip(kernels,biases):
		q = [kernel,bias]
		z,a = conv2d( input ,  q , stride , padding , non_linearialty)
		output_a.append(a)
		output_z.append(z)
		#print('\t\tConvolving with kernel {0} done'.format(i))
		i = i + 1

	output_a = np.array(output_a)
	output_z = np.array(output_z)

	reshaped_out_a=np.zeros((output_a.shape[1],output_a.shape[2],output_a.shape[0]))
	for i in range(output_a.shape[0]):
		for j in range(output_a.shape[1]):
			for k in range(output_a.shape[2]):
				reshaped_out_a[j][k][i]=output_a[i][j][k]

	reshaped_out_z=np.zeros((output_z.shape[1],output_z.shape[2],output_z.shape[0]))
	for i in range(output_z.shape[0]):
		for j in range(output_z.shape[1]):
			for k in range(output_z.shape[2]):
				reshaped_out_z[j][k][i]=output_z[i][j][k]

	return output_a , output_z ,  np.array(reshaped_out_z) , np.array(reshaped_out_a)

####################### Pool Layer #########################

def poollayer( input , type_pool , pool_size):
	output = []
	i = 1
	for i in range(input.shape[0]):
		#print(layer.shape)
		output.append( pooling(input[i],type_pool,pool_size) )
		#print('\t\tpooling of activation Map {0} done'.format(i))
		i = i + 1
		#print(np.array(pooling(layer,type_pool,pool_size)).shape)

	output = np.array(output)
	reshaped_out=np.zeros((output.shape[1],output.shape[2],output.shape[0]))
	for i in range(output.shape[0]):
		for j in range(output.shape[1]):
			for k in range(output.shape[2]):
				reshaped_out[j][k][i]=output[i][j][k]
	return  np.array(reshaped_out)


################ Composition of Convolutional Layers #################

def comp_conv_layer(input_image,num_conv,kernels,strides,paddings,non_linearities,type_pools,pool_size):

	activations = []
	activations_plot = []
	out_pools = []
	input = input_image
	for conv_layer in range(num_conv):
		print('\tConvolutional Layer Initialised')
		print('\tconv layer {0} begin'.format(conv_layer+1))
		activation_plot , activation_ff  = convlayer( input , kernels[conv_layer] , strides[conv_layer] , paddings[conv_layer] , non_linearities[conv_layer])
		activations_plot.append(np.array(activation_plot))
		print('\tconv layer {0} end'.format(conv_layer))
		activations.append( activation_ff )
		print('\tpool layer {0} begin'.format(conv_layer+1))
		p,q = poollayer( activations_plot[-1] , type_pool = type_pools[conv_layer] , pool_size = pool_size )
		out_pools.append( q )
		print('\tpool layer {0} end'.format(conv_layer))
		input = out_pools[-1]

	return np.array(activations_plot)


###################### Unraveling Function #########################

def unravel(input):

	return np.ravel(input)



###################### Multi-Layer Perceptron #########################

def mlp(input,weights,biases,num_hidden,sizes,non_linearialty,output_size):
	input_size = len(input)
	#num_hidden = num_hidden +1
	input = np.array(input).T
	output = np.zeros((output_size))
	
	outputs = []
	
	
	


	for i in range(num_hidden+1):
		output = np.dot(weights[i],input) + biases[i]
		input = output
		outputs.append(output)
	activations = []
	a = []
	#if non_linearialty == 'ReLu':
		
	if non_linearialty == 'sigmoid':
		for output in outputs:
			for unit in output:
				a.append(sigmoid(unit))
			activations.append(a)
			a = []


	if non_linearialty == 'softmax':
		for output in outputs:
			sum = np.sum(output)
			for unit in output:
				a.append(unit/sum)
			activations.append(a)
			a = []
			sum = 0



	return np.array(outputs),np.array(activations)


################## Composition of conolutional Layers and ANN's ################

def CNN(input , num_conv , kernels , strides , paddings , non_linearities , type_pools ,pool_size, num_hidden , sizes , non_linearialty , output_size ):

	conv_out = comp_conv_layer(input,num_conv,kernels,strides,paddings,non_linearities,type_pools,pool_size)
	unraveled_units = unravel(conv_out[-1])

	return mlp( unraveled_units , num_hidden,sizes,non_linearialty,output_size )








