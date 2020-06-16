# Author: Steven Zhao
# Coding: utf-8
# Simple Genetic Algorithm
###################################################################

import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Gene:
    """
    Class to represent individual gene in the algorithm.
    """
    def __init__(self, data=[], fitness=float('inf')):
        self.data = data
        self.size = len(data)
        self.fitness = fitness

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data
        self.size = len(data)

    def get_size(self):
        return self.size

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness


class GeneticAlgorithm:
    """
    Class of genetic algorithm.
    """
    def __init__(self, func, boundary,
                 population_size=100,
                 generation=200,
                 cross_rate=0.8,
                 mutate_rate=0.05,
                 selection='tournament',
                 tournament_size=3,
                 crossover='one_point',
                 tol=1e-5,
                 plot=False):

        self.population_size = population_size
        self.generation = generation
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.boundary = boundary
        self.population = []
        self.selection_method = selection
        self.tournament_size = tournament_size
        self.crossover_method = crossover
        self.fitness_func = func
        self.tol = tol

        self.generate_plot = plot
        self.ims = []

        # Initialize the population based on boundary.
        while len(self.population) < self.population_size:
            value = []
            for (lower, upper) in boundary:
                value.append(random.uniform(lower, upper))

            # Calculate fitness.
            fitness = self.evaluate(value)

            self.population.append(Gene(data=value, fitness=fitness))

        self.best = self.select_best(self.population)

    def evaluate(self, params):
        return abs(self.fitness_func(params))

    def select_best(self, population):
        return sorted(population, key=lambda x: x.get_fitness(), reverse=False)[0]

    def selection(self, population, k):
        """
        Select k genes based on certain rules. Accepted rules:
        - roulette-wheel
        - tournament
        - stochastic
        - truncation (To DO)
        :param population: The population to select from.
        :param k: Number of genes to select.
        :return: list of genes.
        """
        ret = []

        # Calculate accumulated normalized fitness.
        fitness = [x.get_fitness() for x in population]
        reverse_fitness = [1 / fit for fit in fitness]
        normalized_fitness = [f / sum(reverse_fitness) for f in reverse_fitness]
        for i in range(1, len(fitness)):
            normalized_fitness[i] += normalized_fitness[i - 1]

        if self.selection_method == "roulette_wheel":
            # Selection based on fitness.
            for j in range(k):
                r = random.random()
                for index in range(len(fitness)):
                    if normalized_fitness[index] >= r:
                        ret.append(population[index])
                        break

        if self.selection_method == "stochastic":
            for j in range(k):
                candidates = []
                for _ in range(2):
                    r = random.random()
                    for index in range(len(fitness)):
                        if normalized_fitness[index] >= r:
                            candidates.append(population[index])
                            break
                    candidates.sort(key=lambda x: x.get_fitness(), reverse=False)
                    ret.append(candidates[0])

        if self.selection_method == "tournament":
            for j in range(k):
                candidates = random.choices(self.population, k=self.tournament_size)
                candidates.sort(key=lambda x: x.get_fitness(), reverse=False)
                ret.append(candidates[0])

        return ret

    def crossover(self, gene1, gene2):
        """
        Crossover between two genes and return the offsprings if they are better.
        Methods accepted:
        - one_point
        - two-points (To DO)
        :param gene1: First gene
        :param gene2: Second gene
        :return: offspring1, offspring2
        """
        dim = len(gene1.get_data())
        gene1_data = gene1.get_data()
        gene2_data = gene2.get_data()

        # Crossover at single point to create two offsprings.
        if self.crossover_method == "one_point":
            pos = random.randrange(1, dim)
            offspring1_data = gene1_data[:pos] + gene2_data[pos:]
            offspring2_data = gene2_data[:pos] + gene1_data[pos:]

            offspring1 = Gene(data=offspring1_data)
            offspring2 = Gene(data=offspring2_data)

            offspring1.set_fitness(self.evaluate(offspring1.get_data()))
            offspring2.set_fitness(self.evaluate(offspring2.get_data()))

        return offspring1, offspring2

    def mutation(self, gene):
        """
        Random mutation of a gene
        :param gene: Gene to mutate
        :return: New Gene after mutation.
        """
        # Choose the position for mutation.
        pos = random.randrange(gene.get_size())
        data = gene.get_data()

        # Update new value within the boundary.
        lower, upper = self.boundary[pos]
        new_val = random.uniform(lower, upper)
        data[pos] = new_val
        return Gene(data=data, fitness=self.evaluate(data))

    def population_info(self, population):
        """
        Print and return population statistic info for the current population.
        :param population:
        :return:
        """
        fits = [gene.get_fitness() for gene in population]

        length = len(fits)
        mean = sum(fits) / length
        square_sum = sum(x**2 for x in fits)
        std = abs(square_sum/length - mean**2)**0.5

        # Update best resolution.
        best_ind = self.select_best(population)
        if best_ind.get_fitness() < self.best.get_fitness():
            self.best = best_ind

        print("Best individual found so far is %s, with fitness %.5f" % (self.best.get_data(), self.best.get_fitness()))
        print("Min fitness of current pop: %.4f" % min(fits))
        print("Max fitness of current pop: %.4f" % max(fits))
        print("Avg fitness of current pop: %.4f" % mean)
        print("Std of current pop: %.4f" % std)

        return mean, std, min(fits), max(fits)

    def plot(self):
        x = [p.data[0] for p in self.population]
        y = [p.data[1] for p in self.population]
        im = plt.scatter(x, y, c='blue', marker='.', alpha=0.5)
        plt.xlim(self.boundary[0])
        plt.ylim(self.boundary[1])
        self.ims.append([im])

    def fit(self):
        """
        Main frame for genetic algorithm.
        :return:
        """
        cur_generation = 1
        print("Initial population generated.")

        # Begin evolution.
        while cur_generation <= self.generation:
            print("-- Generation %d --" % cur_generation)

            # Select the base for next generation.
            selected_population = self.selection(self.population, self.population_size)

            next_gen = []
            while len(selected_population) > 1:
                gene1 = selected_population.pop()
                gene2 = selected_population.pop()

                if random.random() < self.cross_rate:               # Crossover.
                    offsprings = self.crossover(gene1, gene2)
                    for offspring in offsprings:
                        if random.random() < self.mutate_rate:      # Mutation
                            gene = self.mutation(offspring)
                            next_gen.append(gene)                   # Add to next generation
                        else:
                            next_gen.append(offspring)
            if selected_population:
                next_gen.extend(selected_population)                # Add the last one gene to next gen if needed.

            # Replace old population with new.
            self.population = next_gen

            # Early Stopping
            if self.best.get_fitness() < self.tol:
                break

            if self.generate_plot:
                self.plot()

            cur_generation += 1
            _ = self.population_info(self.population)

        if self.generate_plot:
            fig = plt.figure()
            ani = animation.ArtistAnimation(fig, self.ims, interval=500, blit=True,
                                            repeat_delay=1000)
            plt.show()