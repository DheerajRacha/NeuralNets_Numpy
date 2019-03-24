import numpy as np
import functions 
import backprop
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
import cv2

data = input_data.read_data_sets('MNIST-data/', one_hot=True)
X_train , X_test = [] , []
for img in data.train.images:
	X_train.append(np.array(img).reshape((28,28)))
for img in data.test.images:
	X_test.append(np.array(img).reshape((28,28)))
Y_train = data.train.labels
Y_test = data.test.labels

print("MNIST-Data Extraced")

############ Defining Variables ####################################
Leraning_rate = 1e-4
epochs  = 5

########### Initialising weights and biases of Conv Layers ##############
num_kernels_1 = 4
kernel_size = 5
stride_1 = 2
pool_size_1 = 1
kernel_1 = []
for i in range(num_kernels_1):
	kernel_1.append( np.random.uniform(0,0.1,size=[kernel_size,kernel_size,1]) )
biases_1 = np.random.uniform(0,0.01,size=[num_kernels_1])
Kernels_1 = [kernel_1,biases_1]

num_kernels_2 = 8
kernel_size = 5
stride_2 = 2
pool_size_2 = 2	
kernel_2 = []
for i in range(num_kernels_2):
	kernel_2.append( np.random.uniform(0,0.1,size=[kernel_size,kernel_size,num_kernels_1]) )
biases_2 = np.random.uniform(0,0.01,size=[num_kernels_2])
Kernels_2 = [kernel_2,biases_2]

############# Initializing Weights and Biases for MLP ################

con_out_size = (28 - kernel_size)/stride_1 + 1
con_out_size = (con_out_size)/pool_size_1
con_out_size = (con_out_size - kernel_size)/stride_2 + 1
con_out_size = (con_out_size)/pool_size_2

num_hidden = 1
output_size = 10
sizes = [1024]
sizes.append(output_size)
weights = []
biases = []
x = int(con_out_size*con_out_size*num_kernels_2)
y = sizes[0]
for i in range(num_hidden+1):
		weight = np.random.uniform(0,0.1,size=[y,x])
		biase= np.random.uniform(0,0.01,size=[y])
		weights.append(weight)
		biases.append(biase)
		if (i+2)>len(sizes):
			break
		else:
			x = sizes[i]
			y = sizes[i+1]

################# Training  #################################

for epoch in range(epochs):
	count = 0
	for x_train,y_train in zip(X_train[:1000],Y_train[:1000]):

		out_a_pool1 ,out_z_pool1, z1 , activation1 =  functions.convlayer( input = x_train , Kernels = Kernels_1 , stride = stride_1 , padding = 0 , non_linearialty = 'ReLu')
		activ_pool1 = functions.poollayer( input = out_a_pool1 , type_pool = 'max' , pool_size = pool_size_1)
		z_pool1 = functions.poollayer( input = out_z_pool1 , type_pool = 'max' , pool_size = pool_size_1)

		out_a_pool2 ,out_z_pool2, z2 , activation2 =  functions.convlayer( input = activ_pool1   ,  Kernels = Kernels_2 , stride = stride_2, padding = 0 , non_linearialty = 'ReLu' )
		activ_pool2 = functions.poollayer( input = out_a_pool2 , type_pool = 'max' , pool_size = pool_size_2)
		z_pool2 = functions.poollayer( input = out_z_pool2 , type_pool = 'max' , pool_size = pool_size_2)

		zs,logits = functions.mlp(input = np.ravel(activ_pool2),weights = weights,biases = biases,num_hidden = num_hidden,sizes = sizes,non_linearialty = 'sigmoid',output_size = output_size)
		logits[-1] = functions.softmax(zs[-1])


		delta = backprop.cost_derivative(logits[-1],y_train)*backprop.softmax_grad(zs[-1])

		der_beta = []
		for i in range(sizes[0]):
			der_beta.append(delta*logits[-2][i])
		der_beta = np.transpose(np.array(der_beta))

		q=0
		for i in range(10):
			q += delta[i]*weights[-1][i]
		delta0 = backprop.sigmoid_prime( zs[0])*q

		der_alpha = []
		for i in range(con_out_size*con_out_size*num_kernels_2):
			der_alpha.append(delta0*( np.ravel(activ_pool2))[i])
		der_alpha = np.transpose(np.array(der_alpha))

		q=0
		for i in range(sizes[0]):
			q += delta0[i]*weights[0][i]
		delta_prime = backprop.sigmoid_prime(np.ravel(activ_pool2))*q
		delta1 = np.reshape(delta_prime ,(con_out_size,con_out_size,num_kernels_2))

		Delta1 = backprop.backpropagate_maxpool(delta = delta1 , current_activation = activ_pool2 , prev_activations = activation2  , pool_size = pool_size_2 )
		a1,b1,c1 = backprop.backpropagate_cnn(loc = 0 , delta = Delta1, current_activation = activation2, prev_activations = activ_pool1 , current_Kernels = Kernels_2 , prev_zs = z_pool1, stride_length = stride_2)

		Delta2 = backprop.backpropagate_maxpool(delta = c1 , current_activation = activ_pool1 , prev_activations = activation1  , pool_size = pool_size_1 )
		a2,b2,c2 = backprop.backpropagate_cnn(loc = 1 , delta = Delta2 , current_activation = activation1 , prev_activations =  X_train[0] , current_Kernels = Kernels_1 , prev_zs = X_train[0] , stride_length = stride_1)

		kernel_1 = kernel_1 - Leraning_rate*a2
		biases_1 = biases_1 - Leraning_rate*np.reshape(b2,(num_kernels_1,))
		Kernels_1 = [kernel_1,biases_1]

		kernel_2 = kernel_2 - Leraning_rate*a1
		biases_2 = biases_2 - Leraning_rate*np.reshape(b1,(num_kernels_2,))
		Kernels_2 = [kernel_2,biases_2]

		weights[0] = weights[0] - Leraning_rate*der_alpha
		biases[0] = biases[0] - Leraning_rate*delta0
		weights[-1] = weights[-1] - Leraning_rate*der_beta
		biases[-1] = biases[-1] - Leraning_rate*delta
		count += 1
		if count%1000 == 0:
			print("Epoch - {} , input index - {}".format(epoch,count))



########### Evaluation ###############
	test_accr = 0
	for x_test,y_test in zip(X_test[:100],Y_test[:100]):
		out_a_pool1 ,out_z_pool1, z1 , activation1 =  functions.convlayer( input = x_test , Kernels = Kernels_1 , stride = stride_1 , padding = 0 , non_linearialty = 'ReLu')
		activ_pool1 = functions.poollayer( input = out_a_pool1 , type_pool = 'max' , pool_size = pool_size_1)
		z_pool1 = functions.poollayer( input = out_z_pool1 , type_pool = 'max' , pool_size = pool_size_1)

		out_a_pool2 ,out_z_pool2, z2 , activation2 =  functions.convlayer( input = activ_pool1   ,  Kernels = Kernels_2 , stride = stride_2, padding = 0 , non_linearialty = 'ReLu' )
		activ_pool2 = functions.poollayer( input = out_a_pool2 , type_pool = 'max' , pool_size = pool_size_2)
		z_pool2 = functions.poollayer( input = out_z_pool2 , type_pool = 'max' , pool_size = pool_size_2)

		zs,logits = functions.mlp(input = np.ravel(activ_pool2),weights = weights,biases = biases,num_hidden = num_hidden,sizes = sizes,non_linearialty = 'sigmoid',output_size = output_size)
		result = functions.softmax(zs[-1])

		if np.argmax(y_test) == np.argmax(result):
			test_accr = test_accr + 1.0
	test_accr = (test_accr/len(Y_test[:100]))*100
	print("Test accuracy",test_accr)

	train_accr = 0
	for x_test,y_test in zip(X_train[:100],Y_train[:100]):
		out_a_pool1 ,out_z_pool1, z1 , activation1 =  functions.convlayer( input = x_test , Kernels = Kernels_1 , stride = stride_1 , padding = 0 , non_linearialty = 'ReLu')
		activ_pool1 = functions.poollayer( input = out_a_pool1 , type_pool = 'max' , pool_size = pool_size_1)
		z_pool1 = functions.poollayer( input = out_z_pool1 , type_pool = 'max' , pool_size = pool_size_1)

		out_a_pool2 ,out_z_pool2, z2 , activation2 =  functions.convlayer( input = activ_pool1   ,  Kernels = Kernels_2 , stride = stride_2, padding = 0 , non_linearialty = 'ReLu' )
		activ_pool2 = functions.poollayer( input = out_a_pool2 , type_pool = 'max' , pool_size = pool_size_2)
		z_pool2 = functions.poollayer( input = out_z_pool2 , type_pool = 'max' , pool_size = pool_size_2)

		zs,logits = functions.mlp(input = np.ravel(activ_pool2),weights = weights,biases = biases,num_hidden = num_hidden,sizes = sizes,non_linearialty = 'sigmoid',output_size = output_size)
		result = functions.softmax(zs[-1])

		if np.argmax(y_test) == np.argmax(result):
			train_accr = train_accr + 1.0
	train_accr = (train_accr/len(Y_train[:100]))*100
	print("Train accuracy",train_accr)

	shuff = list(zip(X_train[:1000],Y_train[:1000]))
	random.shuffle(shuff)
	X_train,Y_train = zip(*shuff)

	shuff = list(zip(X_test[:100],Y_test[:100]))
	random.shuffle(shuff)
	X_test,Y_test = zip(*shuff)









