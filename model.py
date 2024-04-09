import os
import copy
import threading
import numpy as np
from activationLayer import ActivationLayer

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
        models_lock = threading.Lock()
        self.train_model(models_lock)

    def set_arguments_as_class_attributes(self, x_train, y_train, total_epochs, learning_rate, use_multiple_threads):
        self.x_train = x_train
        self.y_train = y_train
        self.total_epochs = total_epochs
        self.learning_rate = learning_rate
        self.num_threads = min(len(x_train), os.cpu_count()
                                ) if use_multiple_threads else 1
        self.samples_per_thread = len(x_train) // self.num_threads

    def train_model(self, models_lock):
        for i in range(self.total_epochs):
            thread_models = [None] * self.num_threads  # Shared resource
            avg_error = [0, 0]  # Shared resource
            self.create_worker_threads(thread_models, avg_error, models_lock)
            Model.calculate_average_error(avg_error, i+1, self.total_epochs)
            self.combine_worker_thread_models(thread_models)

    def create_worker_threads(self, thread_models, avg_error, models_lock):
        threads = []
        for thread in range(self.num_threads):
            t = threading.Thread(target=Model.updateModelParams, args=(
                copy.deepcopy(self), thread_models, thread, avg_error, models_lock))
            threads.append(t)
            t.start()

        for thread in threads:
            thread.join()

    @staticmethod
    def updateModelParams(model, thread_models, thread, avg_error, models_lock):
        local_error = 0
        for j in range(model.samples_per_thread):
            current_sample = j + (thread * model.samples_per_thread)
            output = model.x_train[current_sample]
            for layer in model.layers:
                output = layer.forward_propagation(output)
            # compute loss (for display purpose only)
            local_error += model.loss(model.y_train[current_sample], output)
            # backward propagation
            error = model.loss_prime(model.y_train[current_sample], output)
            for layer in reversed(model.layers):
                error = layer.backward_propagation(error, model.learning_rate)

        with models_lock:
            thread_models[thread] = model
            avg_error[0] += local_error  # Update total error sum
            # Update total sample count
            avg_error[1] += model.samples_per_thread

    @staticmethod
    def calculate_average_error(avg_error, current_epoch, total_epochs):
        avg_error[0] /= avg_error[1]
        print('epoch %d/%d   error=%f' %
              (current_epoch, total_epochs, avg_error[0]))

    def combine_worker_thread_models(self, thread_models):
        self.layers = thread_models[0].layers
        self.sumWorkerThreadLayers(thread_models)
        self.averageWorkerThreadLayers()

    def sumWorkerThreadLayers(self, thread_models):
        for model in thread_models[1:]:
            for layer_index in range(len(self.layers)):
                if type(self.layers[layer_index]) == ActivationLayer:
                    continue
                self.layers[layer_index].weights = np.add(
                    self.layers[layer_index].weights, model.layers[layer_index].weights)
                self.layers[layer_index].bias = np.add(
                    self.layers[layer_index].bias, model.layers[layer_index].bias)

    def averageWorkerThreadLayers(self):
        for layer in self.layers:
            if type(layer) == ActivationLayer:
                continue
            layer.weights = np.divide(layer.weights, self.num_threads)
            layer.bias = np.divide(layer.bias, self.num_threads)
            