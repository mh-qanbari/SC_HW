"""
# Author : Mohammadhossein Qanbari
# St. No. : 830595021
# Genetic Algorithm for Travelling Salesman Problem
# Dataset: ali535, d2103, bayg29
# Selection Type: Tournament Selection
# Crossover Type: Rotational Crossover
# Mutation Type: Insertion Mutation
"""

from random import shuffle
from math import radians, sin, cos, asin, sqrt

# <editor-fold desc="Parameters">
g_POPULATION_SIZE = 50
g_CROSSOVER_START = 0
g_CROSSOVER_END = 10

g_ALI535_FILENAME = "ali535.tsp"
g_SELECTED_DATASET_FILE = g_ALI535_FILENAME
g_DS_FEATURE_NAME = "NAME"
g_DS_FEATURE_DIMENSION = "DIMENSION"
g_DS_FEATURE_BEGIN_DATA = "NODE_COORD_SECTION"
g_DS_FEATURE_EOF = "EOF"
g_EARTH_RADIUS = 6371  # kilometers
false = False
true = True
# </editor-fold>


class City:
    def __init__(self):
        self.name_c = str(None)
        self.coord_lon = None
        self.coord_lat = None

    def __str__(self):
        return "City[%s] : coord = (%.2f, %.2f)" % (self.name_c, self.coord_lat, self.coord_lon)

    def get_distance(self, another):
        lat1, lon1, lat2, lon2 = map(radians, [self.coord_lat, self.coord_lon, another.coord_lat, another.coord_lon])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return c * g_EARTH_RADIUS

    @staticmethod
    def compute_distance(source, destination):
        lat1, lon1, lat2, lon2 = map(radians, [source.coord_lat, source.coord_lon, destination.coord_lat, destination.coord_lon])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return c * g_EARTH_RADIUS


# <editor-fold desc="The world of genetic algorithm.">
'''
# Dataset: This class contains the world of a special problem and necessary methods of genetic algorithm.
# Fields:   <name_ds, str> : Name of dataset> ,
            <dimension, int> : Number of cities> ,
            <cities, list(City)> : List of world cities. Each city is an abstract object of City class> ,
            <__best_tour, list(City)> : Best tour path until now. This is a private field of class> ,
            <__best_distance, int> : Distance of best tour until now.>
# Methods:
'''
class Dataset:
    def __init__(self):
        # Public:
        self.name_ds = None
        self.dimension = None
        self.cities = list()
        # Private:
        self.__best_tour = list()
        self.__best_distance = None

    # <editor-fold desc="Generate random population">
    def generate_population(self, population_size=g_POPULATION_SIZE):
        tour_list = []
        for i in range(population_size):
            tour_index_index = range(self.dimension)
            tour = [self.cities[index] for index in tour_index_index]
            shuffle(tour)
            tour_list.append(tour)
            self.suggest(tour)
        return tour_list
    # </editor-fold>

    # <editor-fold desc="Evaluate fitness">
    @staticmethod
    def compute_fitness(tour):
        dist = 1. / Dataset.compute_distance(tour)
        return dist

    @staticmethod
    def compute_distance(tour):
        n = len(tour)
        tour_distance = 0
        for i in range(n):
            j = (i + 1) % n
            city1 = tour[i]
            city2 = tour[j]
            tour_distance += City.get_distance(city1, city2)
        return tour_distance

    def get_distance(self):
        if self.__best_tour is None:
            return None
        if self.__best_distance is not None:
            return self.__best_distance
        dist = 0
        for source_index in self.__best_tour:
            dest_index = (source_index + 1) % self.dimension
            source = self.cities[source_index]
            destination = self.cities[dest_index]
            dist += City.compute_distance(source, destination)
        self.__best_distance = dist
        return dist

    def suggest(self, tour):
        dist = Dataset.compute_distance(tour)
        if dist < self.__best_distance:
            self.__best_tour = tour
            self.__best_distance = dist
        return dist

    def get_fitness(self):
        return 1.0 / self.__best_distance
    # </editor-fold>

    # <editor-fold desc="Selection for crossover. <type, Tournament>">
    # Selection
    @staticmethod
    def tournament_selection(population):
        pop_copy = [[city for city in tour] for tour in population]
        while true:
            n = len(pop_copy)
            last_index = n - 1
            second_last_index = n - 2
            winners = []
            for i in range(0, n, 3):
                winner = None
                index1 = i
                index2 = i + 1
                index3 = i + 2
                if index2 > last_index:
                    winner = tour1
                elif index2 > second_last_index:
                    tour1 = pop_copy[index1]
                    tour2 = pop_copy[index2]
                    fitness1 = Dataset.compute_fitness(tour1)
                    fitness2 = Dataset.compute_fitness(tour2)
                    if fitness1 >= fitness2:
                        winner = tour1
                    else:
                        winner = tour2
                else:
                    tour1 = pop_copy[index1]
                    tour2 = pop_copy[index2]
                    tour3 = pop_copy[index3]
                    fitness1 = Dataset.compute_fitness(tour1)
                    fitness2 = Dataset.compute_fitness(tour2)
                    fitness3 = Dataset.compute_fitness(tour3)
                    if fitness1 >= fitness2:
                        if fitness1 >= fitness3:
                            winner = tour1
                        else:
                            winner = tour3
                    else:
                        if fitness2 >= fitness3:
                            winner = tour2
                        else:
                            winner = tour3
                winners.append(winner)
            print "winners count: ", len(winners)
            if len(winners) == 1:
                print "done"
                return winners[0]
            pop_copy = winners
    # Crossover
    def crossover(self, population):
        tour1 = Dataset.tournament_selection(population)
        tour2 = Dataset.tournament_selection(population)
        # child1 =
    # </editor-fold>

    def get_best_tour(self):
        return self.__best_tour

    def __str__(self):
        return ("DataSet.name = %s\nDataSet.Dimesion = %d\nDataSet.Cities: " %
                (self.name_ds, self.dimension)) + "\n[\n\t" + "\n\t".join([str(city) for city in self.cities]) + "\n]"
# </editor-fold>


if __name__ == "__main__":
    ds = Dataset()

    # Load dataset
    with open(g_SELECTED_DATASET_FILE, mode='r') as ds_file:
        read_data = false
        for line in ds_file:
            line = line.rstrip()
            if line == g_DS_FEATURE_EOF:
                break
            words = line.split(':')
            key = words[0].replace(" ", '')
            if read_data:
                city = City()
                words = line.split()
                city.name_c = words[0]
                city.coord_lon = float(words[1])
                city.coord_lat = float(words[2])
                ds.cities.append(city)
            elif key == g_DS_FEATURE_NAME:
                ds.name_ds = words[1].replace(" ", '')
            elif key == g_DS_FEATURE_DIMENSION:
                ds.dimension = int(words[1].replace(" ", ''))
            elif key == g_DS_FEATURE_BEGIN_DATA:
                read_data = true

    # Generate population
    tour_list = ds.generate_population()

    # <editor-fold desc="Crossover">
    # Selection for crossover
    new_generation = []
    for i in range(g_POPULATION_SIZE):
        tour1 = Dataset.tournament_selection(tour_list)
        tour2 = Dataset.tournament_selection(tour_list)

    # </editor-fold>
