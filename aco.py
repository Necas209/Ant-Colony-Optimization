import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

AnyPath = str | bytes | os.PathLike


def load_distances(path: AnyPath) -> NDArray[np.int32]:
    arr = np.loadtxt(
        path,
        dtype=np.int32,
        delimiter='\t',
    )
    return arr


AntPath = NDArray[np.int32]


@dataclass
class ACO:
    distances: NDArray[np.int32]
    cities: NDArray[str]
    alpha: float = 1.0
    beta: float = 1.0
    decay: float = 0.95
    q: float = 1.0
    num_ants: int = 10
    num_iterations: int = 100
    print_frequency: int = 10
    pheromones: NDArray[np.float32] = None
    best_path: AntPath = None
    best_path_distance: float = np.inf

    def __post_init__(self) -> None:
        self.best_path = np.zeros(self.num_cities, dtype=np.int32)
        self.pheromones = np.ones((self.num_cities, self.num_cities))

    @cached_property
    def num_cities(self) -> int:
        return len(self.cities)

    def run(self) -> None:
        for i in range(self.num_iterations):
            self.run_iteration()
            self.update_pheromones()
            if i % self.print_frequency == 0:
                print(f'Iteration {i}: {self.best_path_distance}')

    def run_iteration(self) -> None:
        for ant in range(self.num_ants):
            path = self.generate_path()
            distance = self.calculate_path_distance(path)
            if distance < self.best_path_distance:
                self.best_path_distance = distance
                self.best_path = path

    def generate_path(self) -> AntPath:
        path = np.zeros(self.num_cities, dtype=np.int32)
        path[0] = np.random.randint(self.num_cities)
        for i in range(1, self.num_cities):
            path[i] = self.select_next_city(path[:i])
        return path

    def select_next_city(self, path: AntPath) -> int:
        current_city = path[-1]
        unvisited_cities = np.delete(np.arange(self.num_cities), path)
        next_city = np.random.choice(unvisited_cities, p=self.calculate_probabilities(current_city, unvisited_cities))
        return next_city

    def calculate_probabilities(self, current_city: int, unvisited_cities: NDArray[np.int32]) -> NDArray[np.float32]:
        probabilities = np.zeros(self.num_cities)
        for city in unvisited_cities:
            probabilities[city] = self.calculate_probability(current_city, city)
        probabilities = probabilities[unvisited_cities]
        probabilities /= np.sum(probabilities)
        return probabilities

    def calculate_probability(self, current_city: int, next_city: int) -> float:
        return self.pheromones[current_city][next_city] ** self.alpha * \
            (1.0 / self.distances[current_city][next_city]) ** self.beta

    def calculate_path_distance(self, path: AntPath) -> float:
        distance = 0.0
        for i in range(self.num_cities - 1):
            distance += self.distances[path[i]][path[i + 1]]
        return distance

    def update_pheromones(self) -> None:
        self.pheromones *= self.decay
        for ant in range(self.num_ants):
            for i in range(self.num_cities - 1):
                self.pheromones[i][i + 1] += self.q / self.calculate_path_distance(self.best_path)
        self.pheromones += self.pheromones.T

    def print_results(self) -> None:
        print('Best path:', self.best_path)
        print('Best path (cities):', self.cities[self.best_path])
        print('Best path distance:', self.best_path_distance)
        print('Pheromones:')
        print(self.pheromones)

    def print_cities(self) -> None:
        print('Cities:')
        for i, city in enumerate(self.cities):
            print(f'{i}: {city}')
