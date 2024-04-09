import time
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from model import Model
from fullyConnectedLayer import FCLayer
from activationLayer import ActivationLayer
from activationFunctions import tanh, tanh_prime
from lossFunctions import mse, mse_prime

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Data loaded")
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
y_train = to_categorical(y_train)
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

net = Model()
# input_shape=(1, 28*28) -> output_shape=(1, 100)
net.add(FCLayer(28*28, 100))
net.add(ActivationLayer(tanh, tanh_prime))
# input_shape=(1, 100) -> output_shape=(1, 100)
net.add(FCLayer(100, 100))
net.add(ActivationLayer(tanh, tanh_prime))
# input_shape=(1, 100) -> output_shape=(1, 50)
net.add(FCLayer(100, 50))
net.add(ActivationLayer(tanh, tanh_prime))
# input_shape=(1, 50) -> output_shape=(1, 10)
net.add(FCLayer(50, 10))
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
start_time = time.time()
net.fit(
    x_train[0:500],
    y_train[0:500],
    total_epochs=100,
    learning_rate=0.1,
    use_multiple_threads=True,
)
print("--- %s seconds ---" % (time.time() - start_time))

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])
