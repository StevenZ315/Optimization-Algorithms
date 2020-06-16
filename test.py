from heuristic_algorithm.genetic_algorithm import GeneticAlgorithm
from heuristic_algorithm.pso import PSO
from test_function.single_objective import Ackley, Rastrigin
import math
import time
import random


def func(x):
    return -20*math.exp(-0.2*(0.5*(x**2 + y**2))**0.5) - \
           math.exp(0.5*(math.cos(2*math.pi*x) + math.cos(2*math.pi*y))) + math.e + 20


boundary = [(-10, 10), (-10, 10)]


def eval(algorithm, func, boundary, repeat=100, **kwargs):
    start = time.time()
    for _ in range(repeat):
        alg = algorithm(func, boundary, **kwargs)
        alg.fit()
    t = time.time() - start
    print("Algorithm run %d times.\t Total time: %.2f seconds" % (repeat, t))


#eval(PSO, func, boundary)
eval(PSO, Rastrigin(dim=2).function, boundary, repeat=25, plot=False, population_size=200)