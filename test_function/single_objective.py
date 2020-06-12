#
# Author： Steven Zhao
# Some test functions for single objective optimization, including:
#   - Rastrigin function
#   - Ackley function
#   - Sphere function
#   - Rosenbrock
#
#
##########################################################################

import numpy as np

class Function:
    """
    Function Interface.
    """
    def function(self, params):
        """
        Main body of the function.
        """
        pass

    def boundary(self):
        """
        Boundary for all the parameters.
        """
        pass

    def solution(self):
        """
        Optimal solution.
        """
        pass


class Ackley(Function):
    """
    Implementation based on: https://www.sfu.ca/~ssurjano/ackley.html
    """
    def __init__(self, dim, a=20, b=0.2, c=2*np.pi):
        self.a = a
        self.b = b
        self.c = c
        self.dim = dim

    def function(self, params):
        if len(params) != self.dim:
            raise IndexError("Input parameter dimension does not match function dimension: first %d, second %d"
                             % (len(params), self.dim))

        term1 = -1 * self.a * np.exp(-1 * self.b * np.sqrt((1/self.dim) * sum(map(lambda x: x**2, params))))
        term2 = -1 * np.exp((1/self.dim) * sum(map(lambda x: np.cos(self.c * x), params)))
        return term1 + term2 + self.a + np.exp(1)

    def boundary(self):
        return [(-32.768, 32.768)] * self.dim

    def solution(self):
        return [0] * self.dim

