from heuristic_algorithm.genetic_algorithm import GeneticAlgorithm
from test_function.single_objective import Ackley
import math

ackley = Ackley(dim=5)
boundary = ackley.boundary()

GA = GeneticAlgorithm(ackley.function, boundary, generation=5000)
GA.fit()