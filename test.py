import numpy as np
import matplotlib.pyplot as plt
import itertools
def notest(a, b):
    if a > b:
        a = a - 1
    else:
        a = a
    return a

if __name__ == '__main__':
    coeffs = list(itertools.product([0, 1], repeat=5))
    print coeffs
    coeffs = np.column_stack(coeffs)
    print coeffs


