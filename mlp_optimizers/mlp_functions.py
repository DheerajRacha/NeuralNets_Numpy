import numpy as np


######## Functions ##########

def sigmoid(x):
	return (1.0/(1.0+np.exp(-x)))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(x):
    q = np.exp(x-np.max(x))
    # print(x,np.max(x))
    return q/q.sum()

def softmax_derivative(x,y):
	s = softmax(x)
	if y is not None:
		k = s[np.where(np.array(y) == 1)]
		a = - k * s
		a[np.where(y == 1)] = k * (1 - k)
		return a
	return s * (1 - s)

def ReLu(x):
    return np.maximum(x,0)

def cost_derivative(output_activations, y):
    return -2*(y - output_activations)

def cost_function(activation , y):
	y = np.reshape(y,(10,1))
	# print(np.array(activation).shape,np.array(y).shape)
	return np.sum((y - activation)*(y - activation))

def ReLU_derivative(x):
	x = np.array(x)
	x[x<=0] = 0
	x[x>0] = 1
	return x


########## BackPropagation ################

def backprop( activations_prev , zs_prev , weights , delta , activation_fn , y = None ):

    prev_a = np.array((activations_prev)).reshape(np.array((activations_prev)).size,1)
    delta_calc=np.zeros((delta.shape[0],1))

    delta = np.reshape(delta,(delta.shape[0],))
    delta_calc[:,0]=delta
    
    grad_w = delta_calc @ prev_a.T
    grad_b = np.copy(delta)

    if activation_fn == "softmax":
        prev_delta = (weights.T @ delta_calc).reshape(np.array((zs_prev)).shape) * softmax_derivative(zs_prev,y)
    
    if activation_fn == "sigmoid":
        prev_delta = (weights.T @ delta_calc).reshape(np.array((zs_prev)).shape) * sigmoid_derivative(zs_prev)
    if activation_fn == "ReLu":
    	prev_delta = (weights.T @ delta_calc).reshape(np.array((zs_prev)).shape) * ReLU_derivative(zs_prev)
    
    return grad_w, grad_b, prev_delta


########## Feed Forward Network #################

def feed_forward_network(x ,weights, biases , activation_fn ):
	x = np.reshape(x,(np.array(x).shape[0],1))
	activation = x
	activations = [x]
	zs = []
	for b, w in zip(biases, weights):
		z = np.dot(w,activation) + b
		zs.append(z)

		if activation_fn == "ReLu":
			activation = ReLu(z)
		if activation_fn == "sigmoid":
			activation = sigmoid(z)
			
		# activation = (activation - activation.mean())/(0.01 + activation.var())
		activations.append(activation)
		activations[-1] = softmax(zs[-1])
	return activations[-1]

