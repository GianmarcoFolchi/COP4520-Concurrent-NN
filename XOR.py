import numpy as np
import time 
from model import Model
from fullyConnectedLayer import FCLayer
from activationLayer import ActivationLayer
from activationFunctions import tanh, tanh_prime
from lossFunctions import mse, mse_prime

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Model()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
start_time = time.time()
net.fit(x_train, y_train, total_epochs=1000,
        learning_rate=0.1, use_multiple_threads=True)
print("--- %s seconds ---" % (time.time() - start_time))
# test
out = net.predict(x_train)
print(out)
