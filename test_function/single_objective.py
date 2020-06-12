#
# Author： Steven Zhao
# Some test functions for single objective optimization, including:
#   - Rastrigin (DONE)
#   - Ackley (DONE)
#   - Bukin (DONE)
#   - Sphere (DONE)
#   - Rosenbrock (DONE)
#   - Cross-in-tray (DONE)
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
    Implementation reference: https://www.sfu.ca/~ssurjano/ackley.html
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
        return (0,) * self.dim

    def __str__(self):
        return "Ackley:\tDim = " + str(self.dim) + "\tGlobal Minimum: " + str(self.solution())


class Bukin(Function):
    """
    Implementation reference: https://www.sfu.ca/~ssurjano/bukin6.html
    """
    def function(self, params):
        return 100 * np.sqrt(abs(params[1] - 0.01 * params[0]**2)) + 0.01 * abs(params[0] + 10)

    def boundary(self):
        return [(-15, -5), (-3, 3)]

    def solution(self):
        return -10, 1

    def __str__(self):
        return "Bukin:\tDim = 2\tGlobal Minimum: " + str(self.solution())


class Rastrigin(Function):
    """
    Implementation reference: https://www.sfu.ca/~ssurjano/rastr.html
    """
    def __init__(self, dim):
        self.dim = dim

    def function(self, params):
        return 10 * self.dim + sum(map(lambda x: x**2 - 10 * np.cos(2*np.pi*x), params))

    def boundary(self):
        return [(-5.12, 5.12)] * self.dim

    def solution(self):
        return (0,) * self.dim

    def __str__(self):
        return "Rastrigin:\tDim = " + str(self.dim) + "\tGlobal Minimum: " + str(self.solution())


class Sphere(Function):
    """
    Implementation reference: https://www.sfu.ca/~ssurjano/spheref.html
    """
    def __init__(self, dim):
        self.dim = dim

    def function(self, params):
        return sum(map(lambda x: x ** 2, params))

    def boundary(self):
        return [(-10, 10)] * self.dim

    def solution(self):
        return (0,) * self.dim

    def __str__(self):
        return "Sphere:\tDim = " + str(self.dim) + "\tGlobal Minimum: " + str(self.solution())


class Rosenbrock(Function):
    """
    Implementation reference: https://www.sfu.ca/~ssurjano/rosen.html
    """
    def __init__(self, dim):
        self.dim = dim

    def function(self, params):
        return sum(100*(params[i+1] - params[i]**2)**2 + (params[i] - 1)**2 for i in range(self.dim - 1))

    def boundary(self):
        return [(-5, 10)] * self.dim

    def solution(self):
        return (1,) * self.dim

    def __str__(self):
        return "Rosenbrock:\tDim = " + str(self.dim) + "\tGlobal Minimum: " + str(self.solution())


class CrossIT(Function):
    """
    Implementation reference: https://www.sfu.ca/~ssurjano/crossit.html
    """
    def function(self, params):
        return -0.0001 * (abs(np.sin(params[0]) * np.sin(params[1]) * np.exp(abs(100 - (params[0]**2 + params[1]**2)**0.5/np.pi))) + 1)**0.1 + 2.06262

    def boundary(self):
        return [(-10, 10), (-10, 10)]

    def solution(self):
        return 1.3491, -1.3491

    def __str__(self):
        return "CrossIT:\tDim = 2\tGlobal Minimum: " + str(self.solution())
