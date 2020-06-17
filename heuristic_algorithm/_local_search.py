# Author: Steven Zhao
# Coding: utf-8
# Local Search Algorithms, including:
#   - Hill Climbing
#   - Simulated Annealing
###################################################################

import random
from collections import defaultdict
from ._pso import Particle
import math


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
        self.gbest = None

        # History Info.
        self._history = defaultdict(list)
        self._history['boundary'] = self.boundary

    # Initialization
    def initialization(self):
        self.p = Particle(boundary=self.boundary)
        self.p.fitness = self.evaluate(self.p.pos)
        self.gbest = self.p.pos

        self.update_history()

    def evaluate(self, params):
        return abs(self.fitness_func(params))

    def update_history(self):
        curr_generation_history = defaultdict(list)
        curr_generation_history['solution'].append(self.p.pos)
        curr_generation_history['fitness'].append(self.p.fitness)
        curr_generation_history['solution_best'] = self.gbest
        curr_generation_history['fitness_best'] = self.evaluate(self.gbest)

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

            # Update global optimum.
            if self.p.fitness < self.evaluate(self.gbest):
                self.gbest = self.p.pos
            self.update_history()

            if self.p.fitness < self.tol:
                break

        print("=" * 25)
        print("Global optimum: ", self.gbest, "\tfitness: ", self.evaluate(self.gbest))


class Annealing(HillClimbing):
    """
    Simulated Annealing Algorithm.
    """
    def __init__(self, func, boundary,
                 T_max=1000,
                 T_min=1,
                 max_iteration=1500,
                 rate=0.99):
        super().__init__(func, boundary)

        self.T_max = T_max
        self.T_min = T_min
        self.rate = rate
        self.max_iteration = max_iteration


    def deal(self, prev_pos, new_pos, delta, temp):
        if delta < 0:
            return new_pos
        else:
            prob = math.exp(-delta/temp)
            if prob > random.random():
                return new_pos
            else:
                return prev_pos

    def fit(self):
        self.initialization()
        print("Initial solution generated.")

        # Iteration
        iteration = 0
        temp = self.T_max
        self._history['temp'].append(temp)

        while temp > self.T_min:
            iteration += 1
            print("-- Generation %d --" % iteration)

            this_iteration = self.max_iteration
            if temp < math.sqrt(self.T_max):
                this_iteration *= 10
            if temp < math.pow(self.T_max, 0.33):
                this_iteration *= 10

            for i in range(this_iteration):
                prev_fitness = self.p.fitness
                prev_pos = self.p.pos
                new_pos = prev_pos[:]

                # Generate random move for all dimensions.
                for dim in range(len(new_pos)):
                    delta = random.uniform(-self.step, self.step)
                    # Limit inside the boundary.
                    if self.boundary[dim][0] <= new_pos[dim] + delta <= self.boundary[dim][1]:
                        new_pos[dim] = new_pos[dim] + delta
                    else:
                        new_pos[dim] = new_pos[dim] - delta

                # Accept new result based on calculated probability.
                new_fitness = self.evaluate(new_pos)
                self.p.pos = self.deal(prev_pos, new_pos, new_fitness - prev_fitness, temp)
                self.p.fitness = self.evaluate(self.p.pos)

                # Update global optimum.
                if self.p.fitness < self.evaluate(self.gbest):
                    self.gbest = self.p.pos

                # Early stopping.
                if self.p.fitness < self.tol:
                    break
            self.update_history()
            print("Update new position: ", self.p.pos[:], "\tfitness: ", self.p.fitness)
            temp = temp * math.pow(0.99, iteration)
            self._history['temp'].append(temp)

        print("="*25)
        print("Global optimum: ", self.gbest, "\tfitness: ", self.evaluate(self.gbest))


