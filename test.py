from heuristic_algorithm.genetic_algorithm import GeneticAlgorithm
from test_function import single_objective
import math

func = single_objective.CrossIT()

boundary = func.boundary()

GA = GeneticAlgorithm(func.function, boundary,
                      population_size=1000,
                      generation=1000,
                      selection="tournament",
                      tournament_size=9)
GA.fit()

print(func)