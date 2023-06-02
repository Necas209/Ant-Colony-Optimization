import numpy as np

from aco import ACO, load_distances
from aco2 import AntColony


def main():
    with open('data/cities.txt', encoding='utf8') as f:
        cities = np.array(f.read().splitlines())
    distances = load_distances('data/distances.txt')
    aco = ACO(
        distances=distances,
        cities=cities,
        num_ants=100,
        num_iterations=500,
        print_frequency=50,
        alpha=1.5,
        beta=1.5,
    )
    aco.run()
    aco.print_results()
    # 2nd algorithm
    aco2 = AntColony(
        distances=distances,
        n_ants=100,
        n_iterations=500,
        decay=0.95,
        n_best=5,
    )
    shortest_path, distance = aco2.run()
    print(f'Shortest path: {shortest_path}')
    print(f'Distance: {distance}')


if __name__ == '__main__':
    main()
