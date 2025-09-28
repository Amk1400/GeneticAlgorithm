
from genetic import GeneticAlgorithm
from utils import generate_initial_population, create_distance_matrix, plot_route,read_dataset,fitness
import matplotlib.pyplot as plt


# TSP problem parameters

CITY_COORDINATES = read_dataset("dj38")

NUM_CITIES = len(CITY_COORDINATES)
"""
def tune_and_analyze(population, fitness_func):
    results = []
    mutation_rates = [0.2]#, 0.05, 0.1, 0.01, 0.3, 0.5, 0.8, 1]#0.2
    crossover_rates = [0.2]
    tournament_sizes = [1, 2, 3, 5, 8, 12,20]

    for m in mutation_rates:
        for c in crossover_rates:
            for t in tournament_sizes:
                ga = GeneticAlgorithm(
                    population.copy(), fitness_func,
                    num_generations=300,
                    mutation_rate=m,
                    crossover_rate=c,
                    tournament_size=t
                )
                _, score, _, _ = ga.run()
                results.append({'mutation_rate': m, 'crossover_rate': c, 'tournament_size': t, 'score': score})

    best = max(results, key=lambda x: x['score'])
    print('Best params:', best)
"""

def main():
    # Create a distance matrix based on city coordinates.
    distance_matrix = create_distance_matrix(CITY_COORDINATES)
    # Generate initial population of random routes.
    population = generate_initial_population(num_individuals=250, num_cities=NUM_CITIES)
    # Wrap the fitness function with the distance matrix.
    fitness_func = lambda route: fitness(route, distance_matrix)

    # Initialize the GeneticAlgorithm with all operators as in-class methods.
    ga = GeneticAlgorithm(population, fitness_func,
                          num_generations=600,
                          mutation_rate=0.1,
                          crossover_rate=0.8,
                          tournament_size=3)
    
    #tune_and_analyze(population,fitness_func)
    best_route, best_fitness,fitness_hist,distance_hist = ga.run()
    print("Best route found:", best_route,len(best_route))
    print("Best fitness found:", best_fitness)
    print("Best distance found:", 1/best_fitness)
    plt.plot(fitness_hist, label='Fitness History')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.title('Fitness History')
    plt.legend()
    plt.show()

    plt.plot(distance_hist, label='Distance History')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.title('distance History')
    plt.legend()
    plt.show()
    plot_route(CITY_COORDINATES, best_route)

if __name__ == '__main__':
    main()