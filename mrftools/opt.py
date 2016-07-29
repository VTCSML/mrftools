import numpy as np

def sgd(func, grad, x, args, callback):
    t = 1
    tolerance = 1e-8
    change = np.inf

    max_iter = 200

    while change > tolerance and t < max_iter:
        old_x = x
        g = grad(x, args)
        x = x - 0.1 * g / t
        change = np.sum(np.abs(x - old_x))
        t += 1
        callback(x)

    return x

def ada_grad(func, grad, x, args, callback):
    t = 1
    tolerance = 1e-8
    max_iter = 500
    grad_norm = np.inf

    grad_sum = 0
    while grad_norm > tolerance and t < max_iter:
        func(x, args)
        old_x = x
        g = grad(x, args)
        grad_sum += g * g
        x = x - 0.1 * g / (np.sqrt(grad_sum) + 0.001)
        grad_norm = np.sqrt(g.dot(g))
        t += 1
        if callback:
            callback(x)
    return x


import matplotlib.pyplot as plt
import time

class WeightRecord(object):
    def __init__(self):
        self.weight_record = np.array([])
        self.time_record = np.array([])

    def callback(self, x):
        a = np.copy(x)
        if (self.weight_record.size) == 0:
            self.weight_record = a.reshape((1, a.size))
            self.time_record = np.array([int(round(time.time() * 1000))])
        else:
            self.weight_record = np.vstack((self.weight_record,a))
            self.time_record = np.vstack((self.time_record,int(round(time.time() * 1000))))


class ObjectivePlotter(object):

    def __init__(self, func):
        self.objectives = []
        self.func = func
        plt.figure()
        self.timer = time.time()
        self.interval = 0.5

    def callback(self, x):
        self.objectives.append(self.func(x))

        elapsed_time = time.time() - self.timer

        if elapsed_time > self.interval:

            plt.clf()

            plt.subplot(121)

            plt.plot(self.objectives)
            plt.ylabel('Objective')
            plt.xlabel('Iteration')

            plt.subplot(122)
            plt.plot(x)
            plt.title('Current solution')

            plt.pause(1e-16)

            self.timer = time.time()