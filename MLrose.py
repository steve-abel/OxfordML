
# pip install mlrose in python - if using environments in jupyter you need to make sure 
# 
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import matplotlib.pyplot as plt

# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

# Initialize fitness function object using coords_list
fitness_coords = mlrose.TravellingSales(coords = coords_list)

# Make a pretty plot of any route: where state is a list of cities ordered by visit  
def plot_route(coords_list, state):
    print(state)
    x = [coords_list[city][0] for city in state]
    y = [coords_list[city][1] for city in state]

    #plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'bo')
    plt.plot(x + [x[0]], y + [y[0]], 'r-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Traveling Salesman Problem')
    plt.show()

# Define problem for the mlrose.genetic_alg routine
problem_fit = mlrose.TSPOpt(length = 8, fitness_fn = fitness_coords,
                            maximize=False)

# Now solve TSPproblem using the genetic algorithm
best_state, best_fitness = mlrose.genetic_alg(problem_fit, mutation_prob = 0.2,
                                              max_attempts = 100, random_state = 2)

print('The best state found is: ', best_state)
print('The fitness at the best state is: ', best_fitness)

plot_route(coords_list,best_state)
