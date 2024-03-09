import copy
import threading
import numpy as np
from ActivationLayer import ActivationLayer


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate, numThreads):
        models_lock = threading.Lock()

        for i in range(epochs):
            thread_models = [None] * numThreads  # Shared resource
            avg_error = [0]  # Shared resource

            samples_per_thread = len(x_train) // numThreads
            threads = []
            for thread in range(numThreads):
                t = threading.Thread(target=updateModelParams, args=(
                    copy.deepcopy(self), thread_models, thread, samples_per_thread, x_train, y_train, learning_rate, avg_error, models_lock))
                threads.append(t)
                t.start()

            for thread in threads:
                thread.join()

            # calculate average error on all samples
            avg_error[0] /= len(x_train)
            print('epoch %d/%d   error=%f' % (i+1, epochs, avg_error[0]))

            # combine all models
            self.layers = thread_models[0].layers
            for model in thread_models[1:]:
                for layer_index in range(len(self.layers)):
                    if type(self.layers[layer_index]) == ActivationLayer: 
                        continue
                    self.layers[layer_index].weights = np.add(
                        self.layers[layer_index].weights, model.layers[layer_index].weights)
                    self.layers[layer_index].bias = np.add(
                        self.layers[layer_index].bias, model.layers[layer_index].bias)

            for layer in self.layers:
                if type(self.layers[layer_index]) == ActivationLayer:
                    continue
                layer.weights = np.divide(layer.weights, numThreads)
                layer.bias = np.divide(layer.bias, numThreads)


def updateModelParams(model, thread_models, thread, samples_per_thread, x_train, y_train, learning_rate, avg_error, models_lock):
    err = 0
    for j in range(samples_per_thread):
        # forward propagation
        current_sample = j + (thread * samples_per_thread)
        output = x_train[current_sample]
        for layer in model.layers:
            output = layer.forward_propagation(output)
        # compute loss (for display purpose only)
        err += model.loss(y_train[current_sample], output)
        # backward propagation
        error = model.loss_prime(y_train[current_sample], output)
        for layer in reversed(model.layers):
            error = layer.backward_propagation(error, learning_rate)

    with models_lock:
        thread_models[thread] = model
        avg_error[0] += err
    
