import numpy as np
import copy

def sgd(func, grad, x, args, callback):
    t = 1
    tolerance = 1e-8
    change = np.inf

    max_iter = 100

    while change > tolerance and t < max_iter:
        old_x = x
        g = grad(x, args)
        x = x - 0.1 * g / t
        change = np.sum(np.abs(x - old_x))
        t += 1
        callback(x)

    return x

def ada_grad(func, grad, x, args, callback, iter_num = 500):
    t = 1
    tolerance = 1e-6
    max_iter = iter_num
    grad_norm = np.inf
    grad_sum = 0
    f = func(x, args)
    while grad_norm > tolerance and t < max_iter:
        f = func(x, args)
        old_x = x
        g = grad(x, args)
        grad_sum += g * g
        x = x - 0.1 * g / (np.sqrt(grad_sum) + 0.5)
        grad_norm = np.sqrt(g.dot(g))
        t += 1
        if callback:
            callback(x)
    return x

def ada_grad_1(func, grad, x, zero_index, args, callback, tot_grad):
    t = 1
    tolerance = 1e-6
    max_iter = 1500
    grad_norm = np.inf
    # print(zero_index)
    # x[zero_index] = 0
    # print('start')
    # print(zero_index)
    # tot_grad = np.zeros(len(x))
    grad_sum = 0
    g = grad(x, args)
    # g[zero_index] = 0
    # print('First gradient')
    # print(g)
    # print(np.sqrt(g.dot(g)))
    while grad_norm > tolerance and t < max_iter:
        x[zero_index] = 0
        f = func(x, args)
        # print(f)
        old_x = x
        g = grad(x, args)
        full_grad = copy.deepcopy(g)
        # print('///')
        # print(g)
        # g[zero_index] = 0
        # print(g)
        grad_norm = g * g
        grad_sum += grad_norm
        tot_grad += np.abs(g)
        ##########
        # x = x - 0.1 * g / (np.sqrt(grad_sum) + 0.001)
        ###########
        x = x - (0.1 /(tot_grad + 0.001)) * g

        grad_norm = np.sqrt(g.dot(g))
        t += 1
        if callback:
            callback(x)
    # print(x)
    tot_grad[zero_index] = 0
    print('iter')
    print(t)
    # print('out')
    # print(full_grad)
    # print(f)
    # print('Last gradient')
    # print(g)
    # print(np.sqrt(g.dot(g)))
    return x, tot_grad
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
