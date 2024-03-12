import os
import copy
import numpy as np
import multiprocessing

from activationLayer import ActivationLayer

# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65 
class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, total_epochs, learning_rate, use_multiple_threads=True):
        self.set_arguments_as_class_attributes(
            x_train, y_train, total_epochs, learning_rate, use_multiple_threads)
        error_lock = multiprocessing.Lock()
        self.train_model(error_lock)

    def set_arguments_as_class_attributes(self, x_train, y_train, total_epochs, learning_rate, use_multiple_threads):
        self.x_train = x_train
        self.y_train = y_train
        self.total_epochs = total_epochs
        self.learning_rate = learning_rate
        self.num_threads = min(len(x_train), os.cpu_count()
                               ) if use_multiple_threads else 1
        self.samples_per_thread = len(x_train) // self.num_threads

    def train_model(self, error_lock):
        for i in range(self.total_epochs):
            thread_models = multiprocessing.Queue()
            avg_error = multiprocessing.Array(
                'd', [0.00001, 0.00001])  # Shared resource
            self.create_worker_process(thread_models, avg_error, error_lock)
            Model.calculate_average_error(avg_error, i+1, self.total_epochs)
            self.combine_worker_thread_models(thread_models)

    def create_worker_process(self, thread_models, avg_error, error_lock):
        processes = []
        for i in range(self.num_threads):
            # Maybe modify this to have only the weights and biases and not anything else 
            process = multiprocessing.Process(target=self.updateModelParams, args=( 
                copy.deepcopy(self.layers), thread_models, i, avg_error, error_lock))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    def updateModelParams(self, model_layers, thread_models, thread, avg_error, error_lock):
        local_error = 0
        for j in range(self.samples_per_thread):
            current_sample = j + (thread * self.samples_per_thread)
            output = self.x_train[current_sample]
            for layer in model_layers:
                output = layer.forward_propagation(output)
            # compute loss (for display purpose only)
            local_error += self.loss(self.y_train[current_sample], output)
            # backward propagation
            error = self.loss_prime(self.y_train[current_sample], output)
            for layer in reversed(model_layers):
                error = layer.backward_propagation(error, self.learning_rate)
        
        thread_models.put(model_layers)
        error_lock.acquire()
        try:
            avg_error[0] += local_error
            avg_error[1] += self.samples_per_thread
        finally:
            error_lock.release()

    @staticmethod
    def calculate_average_error(avg_error, current_epoch, total_epochs):
        avg_error[0] /= avg_error[1]
        print('epoch %d/%d   error=%f' %
              (current_epoch, total_epochs, avg_error[0]))

    def combine_worker_thread_models(self, thread_models):
        self.layers = thread_models.get()
        self.sumWorkerThreadLayers(thread_models)
        self.averageWorkerThreadLayers()

    def sumWorkerThreadLayers(self, thread_models):
        while not thread_models.empty():
            curr_layer = thread_models.get()
            for layer_index in range(len(self.layers)):
                if type(self.layers[layer_index]) == ActivationLayer:
                    continue
                self.layers[layer_index].weights = np.add(
                    self.layers[layer_index].weights, curr_layer[layer_index].weights)
                self.layers[layer_index].bias = np.add(
                    self.layers[layer_index].bias, curr_layer[layer_index].bias)

    def averageWorkerThreadLayers(self):
        for layer in self.layers:
            if type(layer) == ActivationLayer:
                continue
            layer.weights = np.divide(layer.weights, self.num_threads)
            layer.bias = np.divide(layer.bias, self.num_threads)
