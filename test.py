
from heuristic_algorithm import GeneticAlgorithm, PSO
from test_function import Ackley, Rastrigin, CrossIT
import math
import time
import matplotlib.pyplot as plt
from util import AnimatedScatter


def eval(algorithm, func, boundary, repeat=100, **kwargs):
    start = time.time()
    for _ in range(repeat):
        alg = algorithm(func, boundary, **kwargs)
        alg.fit()
        history = alg.history()
    t = time.time() - start
    print("Algorithm run %d times.\t Total time: %.2f seconds" % (repeat, t))


# eval(PSO, func, boundary)
# eval(PSO, Rastrigin(dim=2).function, boundary, repeat=25, population_size=200)

def create_animation(history, **graph_kargs):
    anim = AnimatedScatter(history, **graph_kargs)
    anim.save('test.gif')
    plt.show()


func = Ackley(dim=2)
#func = Ackley(dim=2)

alg = GeneticAlgorithm(func.function, func.boundary(), population_size=200)
alg.fit()
history = alg.history()

create_animation(history, func=func, title='GA with 200 population.')
