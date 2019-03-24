import numpy as np
import functions


def ReLU_derivative(x):
	x = np.array(x)
	x[x<=0] = 0
	x[x>0] = 1
	return x

def softmax_grad(softmax):
    s = functions.softmax(softmax)
    return s*(1-s)

def cost_derivative(output_activations, y):
    return -(y/output_activations) 

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

	
def backpropagate_cnn(loc ,delta , current_activation , prev_activations , current_Kernels , prev_zs, stride_length):
	w , b = current_Kernels[0] , current_Kernels[1]
	#print(np.array(w).shape)
	kernel_size = np.array(w).shape[1]
	if loc == 1:
		prev_depth = 1
	elif loc == 0:
		prev_depth = np.array(prev_activations).shape[2]
	current_height , current_width , current_depth = np.array(current_activation).shape[1] , np.array(current_activation).shape[0] , np.array(current_activation).shape[2]
	prev_height , prev_width   = np.array(prev_activations).shape[0] , np.array(prev_activations).shape[1]
	der_w = np.empty_like(w)
	for r in range(current_depth):
	    for t in range(prev_depth):
	        for h in range(kernel_size):
	            for v in range(kernel_size):
	            	if loc == 1:
	            		prev_a_window = prev_activations[v:v+current_height-kernel_size+1:stride_length,h:h+current_width -kernel_size+1:stride_length]
	            	elif loc == 0:
	                	prev_a_window = prev_activations[v:v+current_height-kernel_size+1:stride_length,h:h+current_width -kernel_size+1:stride_length,t]
	                delta_window  =  delta[ v:v+current_height-kernel_size+1:stride_length,h:h+current_width -kernel_size+1:stride_length,r]
	                der_w[r, h, v , t] = np.sum(prev_a_window * delta_window)

	der_b = np.empty((current_depth, 1))
	for r in range(current_depth):
		der_b[r] = np.sum(delta[:,:,r])

	prev_delta = np.zeros_like(prev_activations)
	for r in range(current_depth):
	    for t in range(prev_depth):
	    	w = np.array(w)
	    	kernel = w[r,:,:,t]
	    	for i, m in enumerate(range(0, prev_height - kernel_size + 1, stride_length)):
	    		for j, n in enumerate(range(0, prev_width - kernel_size + 1, stride_length)):
	    			#print(np.array(kernel).shape)
	    			if loc == 0:
	    				prev_delta[ m:m+kernel_size, n:n+kernel_size,t] += kernel * delta[i, j,r]
	    			elif loc == 1:
	    				prev_delta[ m:m+kernel_size, n:n+kernel_size] += kernel * delta[i, j,r]
	prev_delta *= ReLU_derivative(prev_zs)
	return der_w, der_b, prev_delta


def backpropagate_maxpool(delta , current_activation , prev_activations  , pool_size ):
	current_height , current_width , current_depth = np.array(current_activation).shape[1] , np.array(current_activation).shape[0] , np.array(current_activation).shape[2]
	prev_height , prev_width , prev_depth = np.array(prev_activations).shape[0] , np.array(prev_activations).shape[1] , np.array(prev_activations).shape[2]
	prev_delta = np.empty_like(prev_activations)
	for r, t in zip(range(current_depth), range(prev_depth)):
	    for i, m in enumerate(range(0, prev_height, pool_size)):
	        for j, n in enumerate(range(0, prev_width, pool_size)):
	            prev_a_window = prev_activations[ m:m+pool_size, n:n+pool_size , t]
	            max_unit_index = np.unravel_index(prev_a_window.argmax(), prev_a_window.shape)
	            prev_delta_window = np.zeros_like(prev_a_window)
                prev_delta_window[max_unit_index] = delta[ i, j,t]
                prev_delta[ m:m+pool_size, n:n+pool_size,r] = prev_delta_window
	
	return prev_delta











	



