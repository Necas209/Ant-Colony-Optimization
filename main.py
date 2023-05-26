import numpy as np

from aco import ACO, load_distances


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
    )
    aco.run()
    aco.print_results()


if __name__ == '__main__':
    main()
