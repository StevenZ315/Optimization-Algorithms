
from heuristic_algorithm import GeneticAlgorithm, PSO, HillClimbing
from test_function import Ackley, Rastrigin, CrossIT, Sphere
import math
import time
import matplotlib.pyplot as plt
from util import AnimatedScatter


def eval(algorithm, func, boundary, repeat=1, **kwargs):
    start = time.time()
    for _ in range(repeat):
        alg = algorithm(func, boundary, **kwargs)
        alg.fit()
        history = alg.history()
    t = time.time() - start
    print("Algorithm run %d times.\t Total time: %.2f seconds" % (repeat, t))

func = Rastrigin(dim=2)


eval(HillClimbing, func.function, func.boundary())
# eval(PSO, Rastrigin(dim=2).function, boundary, repeat=25, population_size=200)

def create_animation(history, **graph_kargs):
    anim = AnimatedScatter(history, **graph_kargs)
    anim.save('test.gif')
    plt.show()



#func = Ackley(dim=2)
alg = PSO(func.function, func.boundary())
# alg = HillClimbing(func.function, func.boundary())
alg.fit()
history = alg.history()
print()
create_animation(history, func=func, title='GA with 200 population.', contour=True)
