from aco import ACO, load_distances
from aco2 import AntColony


def main():
    with open('data/cities.txt') as f:
        cities = f.read().splitlines()
    distances = load_distances('data/distances.txt')
    aco = ACO(
        distances=distances,
        cities=cities,
        num_ants=100,
        num_iterations=500,
    )
    aco.run()
    aco.print_results()


if __name__ == '__main__':
    main()
