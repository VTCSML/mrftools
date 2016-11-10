import numpy as np

def sgd(func, grad, x, args, callback):
    t = 1
    tolerance = 1e-8
    change = np.inf

    max_iter = 1000

    while change > tolerance and t < max_iter:
        old_x = x
        g = grad(x, args)
        x = x - 0.1 * g / t
        change = np.sum(np.abs(x - old_x))
        t += 1
        if callback:
            callback(x)

    return x

def ada_grad(func, grad, x, args, callback):
    t = 1
    g_tol = 0.01
    x_tol = 1e-6
    eta = 0.1
    offset = 1.0
    max_iter = 10000
    grad_norm = np.inf
    x_change = np.inf

    grad_sum = 0
    while grad_norm > g_tol and x_change > x_tol and t < max_iter:
        if callback:
            callback(x)
        func(x, args)
        g = grad(x, args)
        grad_sum += g * g
        change = eta * g / (np.sqrt(grad_sum) + offset)
        x = x - change

        grad_norm = np.sqrt(g.dot(g))
        x_change = np.sqrt(change.dot(change))

        # grad_norm = np.sqrt(g.dot(g))

        t += 1

    if callback:
        callback ( x )
    return x

def rms_prop(func, grad, x, args, callback):
    t = 1
    g_tol = 0.02
    x_tol = 1e-6
    max_iter = 10000
    grad_norm = np.inf
    x_change = np.inf
    eta = 0.01
    gamma = 0.1
    eps = 1e-8

    avg_sq_grad = np.zeros(len(x))
    grad_sum = 0
    while grad_norm > g_tol and x_change > x_tol and t < max_iter:
        if callback:
            callback(x)
        func(x, args)
        g = grad(x, args)

        avg_sq_grad = avg_sq_grad * gamma + g ** 2 * (1 - gamma)
        change = eta * g / (np.sqrt(avg_sq_grad) + eps)
        x = x - change

        grad_norm = np.sqrt(g.dot(g))
        x_change = np.sqrt(change.dot(change))
        # grad_norm = np.sqrt(g.dot(g))

        t += 1

    if callback:
        callback ( x )
    return x

def adam(func, grad, x, args, callback):
    t = 1
    g_tol = 1e-3
    x_tol = 1e-3
    max_iter = 10000
    grad_norm = np.inf
    x_change = np.inf

    step_size = 0.01
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8

    m = np.zeros(len(x))
    v = np.zeros(len(x))

    while grad_norm > g_tol and x_change > x_tol and t < max_iter:
        if callback:
            callback(x)
        func(x, args)
        g = grad(x, args)

        m = (1 - b1) * g + b1 * m
        v = (1 - b2) * (g ** 2) + b2 * v
        m_hat = m / (1 - b1 ** (t + 1))
        v_hat = v / (1 - b2 ** (t + 1))
        change = step_size * m_hat / (np.sqrt(v_hat) + eps)
        x = x - change

        grad_norm = np.sqrt(g.dot(g))
        x_change = np.sqrt(change.dot(change))

        t += 1
    if callback:
        callback ( x )
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
            self.time_record = np.array([((time.time() ))])
        else:
            self.weight_record = np.vstack((self.weight_record,a))
            self.time_record = np.vstack ( (self.time_record,  time.time ( ) ) )
            # self.time_record = np.vstack((self.time_record,int(round(time.time() ))))


class ObjectivePlotter(object):

    def __init__(self, func, grad=None):
        self.objectives = []
        self.func = func
        plt.switch_backend("MacOSX")
        self.timer = time.time()
        self.interval = 2.0
        self.last_x = 0
        self.grad = grad
        self.t = 0
        self.iters = []

        if self.grad:
            print "Iter\tf(x)\t\t\tnorm(g)\t\t\tdx"

    def callback(self, x):
        elapsed_time = time.time() - self.timer

        if elapsed_time > self.interval:
            self.objectives.append(self.func(x))
            self.iters.append(self.t)

            plt.clf()

            plt.subplot(131)
            plt.plot(self.iters, self.objectives)
            plt.ylabel('Objective')
            plt.xlabel('Iteration')
            plt.title(self.objectives[-1])

            plt.subplot(132)
            plt.plot(self.iters[-50:], self.objectives[-50:])
            plt.ylabel('Objective')
            plt.xlabel('Iteration')
            plt.title("Zoom")

            plt.subplot(133)
            plt.plot(x)
            plt.title('Current solution')

            # print out diagnostic info
            if self.grad:
                g = self.grad(x)
                diff = x - self.last_x
                print "%d\t%e\t%e\t%e" % (self.iters[-1], self.objectives[-1], np.sqrt(g.dot(g)), np.sqrt(diff.dot(diff)))

            plt.pause(1.0 / 120.0)

            self.timer = time.time()

        self.last_x = x
        self.t += 1
