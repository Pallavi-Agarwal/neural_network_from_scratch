import numpy as np
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data('/tmp/mnist.npz')
# print(x_test.shape)
print("BEFORE RESHAPING")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

y_train = y_train.reshape(60000, 1,1)
y_test = y_test.reshape(10000, 1,1)
print('-----------------------------------')
print("AFTER RESHAPE")

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print("SIR THIS IS THE  [0], BASICALLY THIS IS OUR FIRST IMAGE IN ARRAY FORM")
print(y_train[0])

#
net = Network()
net.add(FCLayer(28,10))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(10, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=3, learning_rate=0.01)

# test
out = net.predict(x_train[0])


print(out)
