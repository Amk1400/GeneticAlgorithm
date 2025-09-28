from math import sqrt

import matplotlib.pyplot as plt
import random
from typing import List, Tuple


def read_dataset(filename: str) -> List[Tuple[float, float]]:
    """
    Reads a .tsp file and extracts city coordinates.

    Args:
        filename (str): Name of the file (without path or extension).

    Returns:
        List[Tuple[float, float]]: List of city coordinates as (x, y) tuples.
    """
    with open("data/" + filename + ".tsp", 'r') as f:
        cities: List[Tuple[float, float]] = []
        for line in f:
            parts = line.split()
            if len(parts) == 3 and ":" not in parts:
                cities.append((float(parts[1]), float(parts[2])))
    return cities


def create_distance_matrix(cities: List[Tuple[float, float]]) -> List[List[float]]:
    return [[euclidian_dist(c1, c2) for c1 in cities] for c2 in cities]

def euclidian_dist(city1, city2):
    return sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def calculate_route_distance(route: List[int], distance_matrix: List[List[float]]) -> float:
    return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1)) + distance_matrix[route[-1]][route[0]]

def fitness(route: List[int], distance_matrix: List[List[float]]) -> float:
    return -calculate_route_distance(route, distance_matrix)

def generate_initial_population(num_individuals: int, num_cities: int) -> List[List[int]]:
    return [random.sample(range(num_cities), num_cities) for _ in range(num_individuals)]
def plot_route(cities, route=None):
    plt.figure(figsize=(10, 6))
    
    if route is not None:
        for i in range(len(route)):
            start = cities[route[i]]
            end   = cities[route[(i + 1) % len(route)]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'r-')

        # Highlight start and end nodes
        start_node = cities[route[0]]
        end_node   = cities[route[-1]]
        plt.scatter(*start_node, c='blue', s=150, marker='*', label='Start')
        # plt.text(start_node[0] + 0.1, start_node[1] + 0.02, "Start", fontsize=9, color='green')
        
        plt.scatter(*end_node, c='red', s=150, marker='X', label='End')
        # plt.text(end_node[0] + 0.1, end_node[1] + 0.02, "End", fontsize=9, color='red')

    # Plot all cities and their indices
    xs = [pt[0] for pt in cities]
    ys = [pt[1] for pt in cities]
    plt.scatter(xs, ys, c='blue', marker='o')
    for i, (x, y) in enumerate(cities):
        plt.text(x + 0.02, y + 0.02, str(i), fontsize=9, color='black')

    plt.title("2D Map of Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.show()
