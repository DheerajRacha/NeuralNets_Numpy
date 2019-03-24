import csv
import numpy as np
import mlp_functions
import random

letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
aplha_dict = dict(zip(letters,[ord(c)%32 for c in letters]))

####### Extract Data ##############

X = []
Y = []

with open('letter-recognition.data') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x = []
        for i in range(1,17):
            # print(i)
            x.append(float(row[i]))
        X.append(x)
        y_int = int(aplha_dict.get(row[0])) - 1
        temp = np.zeros(26)
        temp[y_int] = 1
        Y.append(temp)

shuff = list(zip(X,Y))
random.shuffle(shuff)
X,Y = zip(*shuff)

# print(X[2],Y[2])

X_train = X[:16000]
Y_train = Y[:16000]

# print(Y_train[-1])

X_test = X[16000:]
Y_test = Y[16000:]


# f = open('momentum_results','w')

###### Initializing W & b of MLP ############

num_epochs = 30
learning_rate = 0.1
alpha = 0.01

###### Initializing W & b of MLP ############

sizes = [16,128,64,26]
biases = [np.random.normal(0,1,(y, 1)) for y in sizes[1:]]
vel_b = [np.zeros((y, 1)) for y in sizes[1:]]

weights = [np.random.normal(0,1,(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
vel_w = [np.zeros((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]



########## Training ################

for epoch in range(num_epochs):
    print("Epoch {} Started".format(epoch+1))
    # f.write('Epoch: {}\n'.format(epoch+1))

    count_t = 1
    v_w , v_b = [] , []
    for x_train,y_train in zip(X_train,Y_train):

        x = np.reshape(x_train,(np.array(x_train).shape[0],1))
        y_train = np.reshape(y_train,(np.array(y_train).shape[0],1))

        activation = x
        activations = [x] 
        zs = [x] 
        for b, w in zip(biases, weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = mlp_functions.ReLu(z)
            # activation = (activation - activation.mean())/(0.01 + activation.var())
            activations.append(activation)
        activations[-1] = mlp_functions.softmax(zs[-1])

        delta = -(y_train - activations[-1])
        delta = np.reshape(delta,(26,))

        for j in range(2):
            if(j==0):
                grad_w, grad_b, prev_delta = mlp_functions.backprop(activations[-j-2],zs[-j-2],weights[-j-1],delta,"softmax",y_train)
            else:
                grad_w, grad_b, prev_delta = mlp_functions.backprop(activations[-j-2],zs[-j-2],weights[-j-1],delta,"ReLu",y_train)

            grad_b = np.reshape(grad_b , (grad_b.shape[0],1))

            # print(np.array(vel_w[-j-1]).shape)
            vel_w[-j-1] = vel_w[-j-1] - learning_rate*grad_w
            vel_b[-j-1] = vel_b[-j-1] - learning_rate*grad_b

            weights[-j-1] += vel_w[-j-1]
            biases[-j-1] += vel_b[-j-1]
            delta=prev_delta


        # if count_t%3000 == 0:
        #     print('\t\tIn Train_data with iter {}'.format(count_t))
        count_t = count_t + 1

########### Evaluation ###############

    test_accuracy = 0
    for x_test,y_test in zip(X_test,Y_test):
        activation_t = mlp_functions.feed_forward_network(x_test , weights , biases,"ReLu")

        result = activation_t
        if np.argmax(y_test) == np.argmax(result):
            test_accuracy = test_accuracy + 1.0
    test_accuracy = (test_accuracy/len(Y_test))*100
    print("\tTest accuracy: {}".format(test_accuracy))

    train_accuracy = 0
    for x_train_t,y_train_t in zip(X_train,Y_train):
        activation = mlp_functions.feed_forward_network(x_train_t , weights , biases,"ReLu")

        result = activation
        if np.argmax(y_train_t) == np.argmax(result):
            train_accuracy = train_accuracy + 1.0
    train_accuracy = (train_accuracy/len(Y_train))*100
    print("\tTrain accuracy: {}".format(train_accuracy))

    # f.write("{},{}\n\n".format(test_accuracy,train_accuracy))


############## Shuffle Data ###############

    shuff = list(zip(X_train,Y_train))
    random.shuffle(shuff)
    X_train,Y_train = zip(*shuff)


# f.close()