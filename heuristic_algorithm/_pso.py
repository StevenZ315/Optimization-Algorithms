# Author: Steven Zhao
# Coding: utf-8
# Simple Particle Swarm Algorithm
###################################################################

import random
import numpy as np
from collections import defaultdict


class Particle:
    """
    Class to represent individual particle in the algorithm.
    """
    def __init__(self, boundary):
        self.dim = len(boundary)
        self.pos = [random.uniform(x1, x2) for x1, x2 in boundary]
        self.fitness = 0
        self.pbest = self.pos
        self.fbest = float('inf')
        self.vel = [0] * self.dim

class PSO:
    """
    Class of particle swarm algorithm.
    """
    def __init__(self, func, boundary,
                 population_size=20,
                 generation=100,
                 w=0.6,
                 c1=1,
                 c2=1,
                 tol=1e-5):

        self.population_size = population_size
        self.generation = generation
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.tol = tol
        self.boundary = boundary
        self.population = []
        self.fitness_func = func
        self.gbest = None

        # History Info.
        self._history = defaultdict(list)
        self._history['boundary'] = self.boundary

    def initialization(self):
        for _ in range(self.population_size):
            self.population.append(Particle(boundary=self.boundary))
        self.calc_best(self.population)
        self.population_info(self.population)

    def evaluate(self, params):
        return abs(self.fitness_func(params))

    def calc_best(self, population):
        for p in population:
            p.fitness = self.evaluate(p.pos)
            if p.fitness < p.fbest:
                p.fbest = p.fitness
                p.pbest = p.pos
            if not self.gbest or p.fitness < self.evaluate(self.gbest):
                self.gbest = p.pos

        # Generate history for current generation.
        curr_generation_history = defaultdict(list)
        for p in population:
            curr_generation_history['solution'].append(p.pos)
            curr_generation_history['fitness'].append(p.fitness)
        curr_generation_history['solution_best'] = self.gbest
        curr_generation_history['fitness_best'] = self.evaluate(self.gbest)

        # Append to global history.
        for key in curr_generation_history.keys():
            self._history[key].append(curr_generation_history[key])


    def diff_calc(self, list1, list2):
        """
        Helper function to calculate the difference between two vectors.
        :param list1: Input 1
        :param list2: Input 2
        :return: Difference between two lists.
        """
        return [list1[i] - list2[i] for i in range(len(list1))]

    def sum_calc(self, list1, list2):
        """
        Helper function to calculate the sum between two vectors.
        :param list1: Input 1
        :param list2: Input 2
        :return: Sum between two lists.
        """
        return [list1[i] + list2[i] for i in range(len(list1))]

    def multiply(self, list1, multiplier):
        """
        Helper function to multiply a list with float.
        :param list1:
        :param multiplier:
        :return:
        """
        return [x * multiplier for x in list1]

    def in_boundary(self, pos, boundary):
        """
        Helper function to check if a position in within boundary.
        :param pos:
        :param boundary:
        :return:
        """
        for i in range(len(pos)):
            if pos[i] >= boundary[i][1] or pos[i] <= boundary[i][0]:
                return False
        return True

    def update(self, population):
        """
        Calculate and update new position.
        :param population:
        :return:
        """
        for p in population:
            term1 = self.sum_calc(self.multiply(p.vel, self.w,),
                                  self.multiply(self.diff_calc(p.pbest, p.pos), self.c1 * random.random()))
            term2 = self.multiply(self.diff_calc(self.gbest, p.pos), self.c2 * random.random())
            v_new = self.sum_calc(term1, term2)

            pos_new = self.sum_calc(p.pos, v_new)

            # New position need to be within boundary.
            if self.in_boundary(pos_new, self.boundary):
                p.vel = v_new
                p.pos = pos_new
                p.fitness = self.evaluate(p.pos)
                if p.fitness < p.fbest:
                    p.pbest = p.pos
                    p.fbest = p.fitness

    def population_info(self, population):
        """
        Print and return population statistic info for the current population.
        :param population:
        :return:
        """
        fits = [p.fitness for p in population]

        length = len(fits)
        mean = sum(fits) / length
        square_sum = sum(x ** 2 for x in fits)
        std = abs(square_sum / length - mean ** 2) ** 0.5

        print("Best individual found so far is %s, with fitness %.5f"
              % (self.gbest, self.evaluate(self.gbest)))
        print("Min fitness of current pop: %.4f" % min(fits))
        print("Max fitness of current pop: %.4f" % max(fits))
        print("Avg fitness of current pop: %.4f" % mean)
        print("Std of current pop: %.4f" % std)

        # Update global history
        self._history['fitness_summary'].append((mean, std, min(fits), max(fits)))

    def history(self):
        return self._history

    def fit(self):
        """
        Main frame for particle swarm algorithm.
        :return:
        """
        self.initialization()
        print("Initial population generated.")

        # Iteration
        for i in range(self.generation):
            print("-- Generation %d --" % (i+1))
            self.update(self.population)
            self.calc_best(self.population)

            # Print study info.
            self.population_info(self.population)

            # Early stopping.
            if self.evaluate(self.gbest) < self.tol:
                break


