import numpy as np
import multiprocessing
import cProfile

from model import Model
from fullyConnectedLayer import FCLayer
from activationLayer import ActivationLayer
from activationFunctions import tanh, tanh_prime
from lossFunctions import mse, mse_prime


def main():
    profiler = cProfile.Profile()
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    model = Model()
    model.add(FCLayer(2, 3))
    model.add(ActivationLayer(tanh, tanh_prime))
    model.add(FCLayer(3, 1))
    model.add(ActivationLayer(tanh, tanh_prime))
    model.use(mse, mse_prime)
    command = 'model.fit(x_train, y_train, total_epochs=100, learning_rate=0.5, use_multiple_threads=True)'

# Execute profiling
    profiler.runctx(command, globals(), locals())
    profiler.print_stats()
    # model.fit(x_train, y_train, total_epochs=1000,
    #         learning_rate=0.1, use_multiple_threads=True)
    out = model.predict(x_train)
    print(out)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()