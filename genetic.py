import random
from typing import List, Callable, Tuple


class GeneticAlgorithm:
    def __init__(
        self,
        population: List[List[int]],
        fitness_func: Callable[[List[int]], float],
        num_generations: int = 100,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        selection_strategy: str = 'tournament',  # 'tournament', 'roulette', 'rank'
        crossover_strategy: str = 'uniform',  # 'one_point', 'two_point', 'uniform'
        mutation_strategy: str = 'scramble',        # 'swap', 'inversion', 'scramble'
        tournament_size: int = 3,
    ) -> None:
        """
        Initialize the genetic algorithm with population and configuration parameters.

        Args:
            population (List[List[int]]): Initial population of individuals.
            fitness_func (Callable[[List[int]], float]): Function to evaluate the fitness of an individual.
            num_generations (int): Number of generations to run the algorithm.
            mutation_rate (float): Probability of mutation for each individual.
            crossover_rate (float): Probability of crossover between pairs.
            selection_strategy (str): Selection method ('tournament', 'roulette', or 'rank').
            crossover_strategy (str): Crossover method ('one_point', 'two_point', or 'uniform').
            mutation_strategy (str): Mutation method ('swap', 'inversion', or 'scramble').
            tournament_size (int): Size of the tournament in tournament selection.
        """
        self.population: List[List[int]] = population
        self.fitness_func: Callable[[List[int]], float] = fitness_func
        self.num_generations: int = num_generations
        self.mutation_rate: float = mutation_rate
        self.crossover_rate: float = crossover_rate
        self.selection_strategy: str = selection_strategy
        self.crossover_strategy: str = crossover_strategy
        self.mutation_strategy: str = mutation_strategy
        self.tournament_size: int = tournament_size

        # History tracking
        self.best_fitness_history: List[float] = []
        self.best_distance_history: List[float] = []
    def evaluate_population(self) -> List[float]:
        """
        Evaluate the fitness of the current population.

        Returns:
            List[float]: A list containing the fitness value of each individual.
        """
        return [self.fitness_func(chromosome) for chromosome in self.population]

    def selection(self, fitnesses: List[float]) -> int:
        """
        Select an individual index based on the configured selection strategy.

        Args:
            fitnesses (List[float]): List of fitness values for the population.

        Returns:
            int: Index of the selected individual.
        """
        match self.selection_strategy:
            case 'tournament':
                participants = random.sample(range(len(fitnesses)), self.tournament_size)
                return max(participants, key=lambda i: fitnesses[i])
            case 'roulette':
                adj = [f - min(fitnesses) for f in fitnesses]
                return random.choices(range(len(fitnesses)), weights=adj, k=1)[0]
            case 'rank':
                ranks = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])#کم به زیاده
                return random.choices(ranks, weights=[i + 1 for i in range(len(fitnesses))], k=1)[0]
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Perform permutation-preserving crossover between two parents.

        Args:
            parent1 (List[int]): First parent individual.
            parent2 (List[int]): Second parent individual.

        Returns:
            List[int]: Offspring individual produced by crossover.
        """
        if random.random() > self.crossover_rate:#اگه بیشتر از 0.8 اومد ترکیب انجام نشه
            return parent1[:]
        match self.crossover_strategy:
            case 'one_point':
                p = random.randint(1, len(parent1) - 2)
                child = parent1[:p] + [g for g in parent2 if g not in parent1[:p]]
                return child
            case 'two_point':
                a, b = sorted(random.sample(range(1, len(parent1) - 1), 2))
                segment = parent1[a:b]
                rest = [g for g in parent2 if g not in segment]
                return rest[:a] + segment + rest[a:]
            case 'uniform':
                child = []
                for g1, g2 in zip(parent1, parent2):
                    g = g1 if random.random() < 0.5 else g2
                    if g not in child:
                        child.append(g)
                child += [g for g in parent1 if g not in child]
                return child

    def mutation(self, individual: List[int]) -> List[int]:
        """
        Apply mutation to an individual using the configured mutation strategy.

        Args:
            individual (List[int]): Individual to mutate.

        Returns:
            List[int]: Mutated individual.
        """
        if random.random() <= self.mutation_rate:
            i, j = sorted(random.sample(range(len(individual)), 2))
            match self.mutation_strategy:
                case 'swap':
                    individual[i], individual[j] = individual[j], individual[i]
                case 'inversion':
                    individual[i:j] = individual[i:j][::-1]
                case 'scramble':
                    segment = individual[i:j]
                    random.shuffle(segment)
                    individual[i:j] = segment
        return individual
    def run(self) -> Tuple[List[int], float, List[float], List[float]]:
        """
        Run the genetic algorithm for the configured number of generations.

        Returns:
            Tuple containing:
                - List[int]: The best solution found.
                - float: Fitness of the best solution.
                - List[float]: History of best fitness values.
                - List[float]: History of best distance values.
        """
        for _ in range(self.num_generations):
            #print(_)
            fitnesses = self.evaluate_population()
            best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            best_fit = fitnesses[best_idx]
            self.best_fitness_history.append(best_fit)
            self.best_distance_history.append(1/best_fit)

            new_population = []
            while len(new_population) < len(self.population):
                p1 = self.selection(fitnesses)
                p2 = self.selection(fitnesses)
                child = self.crossover(self.population[p1], self.population[p2])
                child = self.mutation(child)
                new_population.append(child)

            self.population = new_population

        final_fitnesses = self.evaluate_population()
        best_idx = max(range(len(final_fitnesses)), key=lambda i: final_fitnesses[i])
        return (
            self.population[best_idx],
            final_fitnesses[best_idx],
            self.best_fitness_history,
            self.best_distance_history,
        )
