import numpy as np
from numpy.random import choice as np_choice

Path = list[tuple[int, int]]


class AntColony:

    def __init__(self, distances: np.ndarray, n_ants: int, n_best: int, n_iterations: int, decay: float,
                 alpha: float = 1.0, beta: float = 1.0) -> None:
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of the best ants who deposit pheromone
            n_iterations (int): Number of iterations
            decay (float): Rate it which pheromone decays.
            alpha (float): exponent on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (float): exponent on distance, higher beta give distance more weight. Default=1
        Example:
            ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)
        """
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self) -> tuple[list, int]:
        all_time_shortest_path = [], np.inf
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths)
            shortest_path: tuple[Path, np.float] = min(all_paths, key=lambda x: x[1])
            if i % 100 == 0:
                print(shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone = self.pheromone * self.decay
        return all_time_shortest_path

    def spread_pheronome(self, all_paths: list[tuple[Path, float]]) -> None:
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:self.n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path: Path) -> float:
        return sum(self.distances[ele] for ele in path)

    def gen_all_paths(self) -> list[tuple[Path, float]]:
        return [(path := self.gen_path(0), self.gen_path_dist(path))
                for _ in range(self.n_ants)]

    def gen_path(self, start: int) -> Path:
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # going back to where we started
        return path

    def pick_move(self, pheromone: np.ndarray, dist: np.ndarray, visited: set[int]) -> int:
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move
