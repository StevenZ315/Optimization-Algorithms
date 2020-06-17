# Author: Steven Zhao
# Coding: utf-8
# Local Search Algorithms, including:
#   - Hill Climbing
#   - Simulated Annealing
###################################################################

import random
from collections import defaultdict
from ._pso import Particle


class HillClimbing:
    """
    Class of Hill Climbing algorithm.
    """
    def __init__(self, func, boundary,
                 step=0.1,
                 tol=1e-5):

        self.step = step
        self.tol = tol
        self.fitness_func = func
        self.boundary = boundary

        # History Info.
        self._history = defaultdict(list)
        self._history['boundary'] = self.boundary

    # Initialization
    def initialization(self):
        self.p = Particle(boundary=self.boundary)
        self.p.fitness = self.evaluate(self.p.pos)

        self.update_history()

    def evaluate(self, params):
        return abs(self.fitness_func(params))

    def update_history(self):
        curr_generation_history = defaultdict(list)
        curr_generation_history['solution'].append(self.p.pos)
        curr_generation_history['fitness'].append(self.p.fitness)
        curr_generation_history['solution_best'] = self.p.pos

        # Append to global history.
        for key in curr_generation_history.keys():
            self._history[key].append(curr_generation_history[key])

    def history(self):
        return self._history

    def fit(self):
        self.initialization()
        print("Initial solution generated.")

        # Iteration
        iteration = 0
        while True:
            iteration += 1
            print("-- Generation %d --" % iteration)

            # Create neighbor and compare.
            neighbor = []
            curr_pos = self.p.pos
            for dim in range(len(curr_pos)):
                # Verify new neighbor is within range
                new_neighbor_one = curr_pos[:dim] + [curr_pos[dim] + self.step] + curr_pos[dim+1:]
                if new_neighbor_one[dim] < self.boundary[dim][1]:
                    neighbor.append(new_neighbor_one)

                new_neighbor_two = curr_pos[:dim] + [curr_pos[dim] - self.step] + curr_pos[dim+1:]
                if new_neighbor_one[dim] > self.boundary[dim][0]:
                    neighbor.append(new_neighbor_two)

            neighbor_fitness = sorted([(n, self.evaluate(n)) for n in neighbor], key=lambda x: x[1])

            if neighbor_fitness[0][1] > self.p.fitness:
                break

            # Update new position.
            self.p.pos = neighbor_fitness[0][0]
            self.p.fitness = neighbor_fitness[0][1]
            print("Update new position: ", self.p.pos[:], "\tfitness: ", self.p.fitness)

            self.update_history()

            if self.p.fitness < self.tol:
                break


