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

def ada_grad(func, grad, x, args, callback, added_edge, iter_num):
    weights = np.ones(len(x))
    if added_edge is not None:
        weights[added_edge] = .9
    t = 1
    g_tolerance = 1e-15
    f_tolerance = 1e-15
    max_iter = iter_num
    grad_norm = np.inf
    grad_sum = 0
    old_f = float('inf')
    f = func(x, args)
    f_change = float('inf')

    while grad_norm > g_tolerance and f_change > f_tolerance and t < max_iter:
        f = func(x, args)
        if f > 10000:
            print('ERORRR!!!!!!')
            print(f)
            print('ERORRR!!!!!!')
        f_change = np.abs(old_f - f)
        old_f = f
        old_x = x
        g = grad(x, args)
        # print('g')
        # print(g)
        grad_sum += g * g
        # x = x - 0.1 * g / (1 + np.sqrt(grad_sum))
        x = x - g / (1e-5 + np.sqrt(grad_sum) * weights) 
        # x = x - g / (1 + t)
        grad_norm = np.sqrt(g.dot(g))
        t += 1
        if callback:
            callback(x)
    print('weights norm')
    print(np.sqrt(x.dot(x)))
    print('iter num')
    print(t)
    print('gradient norm')
    print(grad_norm/len(x))
    print('obj change')
    print(f_change)
    return x

def ada_grad_1(func, grad, x, zero_index, args, callback, tot_grad):
    t = 1
    tolerance = 1e-6
    max_iter = 1500
    grad_norm = np.inf
    grad_sum = 0
    g = grad(x, args)
    while grad_norm > tolerance and t < max_iter:
        x[zero_index] = 0
        f = func(x, args)
        old_x = x
        g = grad(x, args)
        full_grad = copy.deepcopy(g)
        grad_norm = g * g
        grad_sum += grad_norm
        tot_grad += np.abs(g)
        x = x - g/tot_grad
        grad_norm = np.sqrt(g.dot(g))
        t += 1
        if callback:
            callback(x)
    return x, tot_grad

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
