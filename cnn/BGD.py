import numpy as np
import functions 
import backprop
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random

data = input_data.read_data_sets('MNIST-data/', one_hot=True)
X_train , X_test = [] , []
for img in data.train.images:
	X_train.append(np.array(img).reshape((28,28)))
for img in data.test.images:
	X_test.append(np.array(img).reshape((28,28)))
Y_train = data.train.labels
Y_test = data.test.labels

print("MNIST-Data Extraced")

#################### Defining Variables ############################
Leraning_rate = 1e-4
epochs  = 4
batch_size = 32

########## Initialising weights and biases of Conv Layers ################
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
# print(con_out_size)



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

################# Training #################################


for epoch in range(epochs):
	for batch in range(len(X_train[:10*batch_size])/batch_size):
		BDG = 0
		for x_train,y_train in zip(X_train[batch*batch_size:batch*batch_size+batch_size],Y_train[batch_size*batch:batch*batch_size+batch_size]):

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
			# print(len(delta),"#######")
			for i in range(10):
				q += delta[i]*weights[-1][i]
			delta0 = backprop.sigmoid_prime( zs[0])*q
			# print(np.array(delta0).shape)
			der_alpha = []
			for i in range(con_out_size*con_out_size*num_kernels_2):
				der_alpha.append(delta0*( np.ravel(activ_pool2))[i])
			der_alpha = np.transpose(np.array(der_alpha))

			q=0
			for i in range(sizes[0]):
				q += delta0[i]*weights[0][i]
			delta_prime = backprop.sigmoid_prime(np.ravel(activ_pool2))*q
			delta1 = np.reshape(delta_prime ,(con_out_size,con_out_size,num_kernels_2))
			# print('sdads',len(delta_prime) ,len(delta1))
			Delta1 = backprop.backpropagate_maxpool(delta = delta1 , current_activation = activ_pool2 , prev_activations = activation2  , pool_size = pool_size_2 )
			a1,b1,c1 = backprop.backpropagate_cnn(loc = 0 , delta = Delta1, current_activation = activation2, prev_activations = activ_pool1 , current_Kernels = Kernels_2 , prev_zs = z_pool1, stride_length = stride_2)

			Delta2 = backprop.backpropagate_maxpool(delta = c1 , current_activation = activ_pool1 , prev_activations = activation1  , pool_size = pool_size_1 )
			a2,b2,c2 = backprop.backpropagate_cnn(loc = 1 , delta = Delta2 , current_activation = activation1 , prev_activations =  X_train[0] , current_Kernels = Kernels_1 , prev_zs = X_train[0] , stride_length = stride_1)


			if BDG == 0:
				Delta = np.zeros(np.array(delta).shape)
				Delta0 = np.zeros(np.array(delta0).shape)
				Der_aplha = np.zeros(np.array(der_alpha).shape)
				Der_beta = np.zeros(der_beta.shape)
				A2 = np.zeros(np.array(a2).shape)
				B2 = np.zeros(np.array(b2).shape)
				A1 = np.zeros(np.array(a1).shape)
				B1 = np.zeros(np.array(b1).shape)
				BGD = 1

			A2 += a2
			A1 += a1
			B1 += b1
			B2 += b2

			Delta0 += delta0
			Delta += delta
			Der_aplha += der_alpha
			Der_beta += der_beta

		kernel_1 = kernel_1 - Leraning_rate*A2/batch_size
		biases_1 = biases_1 - Leraning_rate*np.reshape(B2,(num_kernels_1,))/batch_size
		Kernels_1 = [kernel_1,biases_1]

		kernel_2 = kernel_2 - Leraning_rate*A1/batch_size
		biases_2 = biases_2 - Leraning_rate*np.reshape(B1,(num_kernels_2,))/batch_size
		Kernels_2 = [kernel_2,biases_2]

		weights[0] = weights[0] - Leraning_rate*Der_aplha/batch_size
		biases[0] = biases[0] - Leraning_rate*Delta0/batch_size
		weights[-1] = weights[-1] - Leraning_rate*Der_beta/batch_size
		biases[-1] = biases[-1] - Leraning_rate*Delta/batch_size
		print("Epoch {} , Batch {}".format(epoch,batch))
	

############ Evaluating ###################
	test_accr = 0
	for x_test,y_test in zip(X_test[:500],Y_test[:500]):
		out_a_pool1 ,out_z_pool1, z1 , activation1 =  functions.convlayer( input = x_test , Kernels = Kernels_1 , stride = stride_1 , padding = 0 , non_linearialty = 'ReLu')
		activ_pool1 = functions.poollayer( input = out_a_pool1 , type_pool = 'max' , pool_size = pool_size_1)
		z_pool1 = functions.poollayer( input = out_z_pool1 , type_pool = 'max' , pool_size = pool_size_1)

		out_a_pool2 ,out_z_pool2, z2 , activation2 =  functions.convlayer( input = activ_pool1   ,  Kernels = Kernels_2 , stride = stride_1, padding = 0 , non_linearialty = 'ReLu' )
		activ_pool2 = functions.poollayer( input = out_a_pool2 , type_pool = 'max' , pool_size = pool_size_2)
		z_pool2 = functions.poollayer( input = out_z_pool2 , type_pool = 'max' , pool_size = pool_size_2)

		zs,logits = functions.mlp(input = np.ravel(activ_pool2),weights = weights,biases = biases,num_hidden = num_hidden,sizes = sizes,non_linearialty = 'sigmoid',output_size = output_size)
		result = functions.softmax(zs[-1])

		if np.argmax(y_test) == np.argmax(result):
			test_accr = test_accr + 1.0
	test_accr = test_accr/len(Y_test[:500])
	print("Test Accuracy",test_accr)

	trian_accr = 0	
	for x_test,y_test in zip(X_train[:500],Y_train[:500]):
		out_a_pool1 ,out_z_pool1, z1 , activation1 =  functions.convlayer( input = x_test , Kernels = Kernels_1 , stride = stride_1 , padding = 0 , non_linearialty = 'ReLu')
		activ_pool1 = functions.poollayer( input = out_a_pool1 , type_pool = 'max' , pool_size = pool_size_1)
		z_pool1 = functions.poollayer( input = out_z_pool1 , type_pool = 'max' , pool_size = pool_size_1)

		out_a_pool2 ,out_z_pool2, z2 , activation2 =  functions.convlayer( input = activ_pool1   ,  Kernels = Kernels_2 , stride = stride_1, padding = 0 , non_linearialty = 'ReLu' )
		activ_pool2 = functions.poollayer( input = out_a_pool2 , type_pool = 'max' , pool_size = pool_size_2)
		z_pool2 = functions.poollayer( input = out_z_pool2 , type_pool = 'max' , pool_size = pool_size_2)

		zs,logits = functions.mlp(input = np.ravel(activ_pool2),weights = weights,biases = biases,num_hidden = num_hidden,sizes = sizes,non_linearialty = 'sigmoid',output_size = output_size)
		result = functions.softmax(zs[-1])

		if np.argmax(y_test) == np.argmax(result):
			trian_accr = trian_accr + 1.0
	trian_accr = trian_accr/len(Y_train[:500])
	print("Train Accuracy",trian_accr)


######## Shuffle Datasets ###########

	shuff = list(zip(X_train,Y_train))
	random.shuffle(shuff)
	X_train,Y_train = zip(*shuff)

	shuff = list(zip(X_test[:500],Y_test[:500]))
	random.shuffle(shuff)
	X_test,Y_test = zip(*shuff)

		






