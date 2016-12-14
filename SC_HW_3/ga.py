"""
# Author : Mohammadhossein Qanbari
# St. No. : 830595021
# Genetic Algorithm for Travelling Salesman Problem
# Dataset: ali535, d2103, bayg29
# Selection Type: Tournament Selection
# Crossover Type: Rotational Crossover
# Mutation Type: Insertion Mutation
"""

import sys
from random import shuffle
from math import radians, sin, cos, asin, sqrt
import random

# <editor-fold desc="Parameters">
g_POPULATION_SIZE = 50

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


# <editor-fold desc="City class">
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
        if source is None or destination is None:
            return sys.maxint
        lat1, lon1, lat2, lon2 = map(radians, [source.coord_lat, source.coord_lon, destination.coord_lat, destination.coord_lon])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return c * g_EARTH_RADIUS
# </editor-fold>

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
        if isinstance(tour, City):
            print 'test'
        n = len(tour)
        tour_distance = 0
        for i in range(n):
            j = (i + 1) % n
            city1 = tour[i]
            city2 = tour[j]
            tour_distance += City.compute_distance(city1, city2)
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
        n = len(population)
        index1 = random.randint(0, n - 1)
        index2 = random.randint(0, n - 1)
        index3 = random.randint(0, n - 1)
        tour1 = population[index1]
        tour2 = population[index2]
        tour3 = population[index3]
        fitness1 = Dataset.compute_fitness(tour1)
        fitness2 = Dataset.compute_fitness(tour2)
        fitness3 = Dataset.compute_fitness(tour3)
        if fitness1 >= fitness2:
            if fitness1 >= fitness3:
                return tour1
            else:
                return tour3
        else:
            if fitness2 >= fitness3:
                return tour2
            else:
                return tour3

    # Crossover
    def crossover(self, parent1, parent2):
        global child1
        global child2
        child1 = [None for _ in range(len(parent1))]
        child2 = [None for _ in range(len(parent2))]
        crossover_point1 = random.randint(0, self.dimension)
        crossover_point2 = random.randint(0, self.dimension)
        if crossover_point1 > crossover_point2:
            temp = crossover_point1
            crossover_point1 = crossover_point2
            crossover_point2 = temp
        for i in range(crossover_point1, crossover_point2 + 1):
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        last_index1 = (crossover_point2 + 1) % self.dimension
        last_index2 = (crossover_point2 + 1) % self.dimension
        for i in range(self.dimension):
            index = (i + crossover_point1) % self.dimension
            city1 = parent1[index]
            city2 = parent2[index]
            if city2 not in child1:
                child1[last_index1] = city2
                last_index1 = (last_index1 + 1) % self.dimension
            if city1 not in child2:
                child2[last_index2] = city1
                last_index2 = (last_index2 + 1) % self.dimension
        return child1, child2
    # </editor-fold>

    def get_best_tour(self):
        return self.__best_tour

    def __str__(self):
        return ("DataSet.name = %s\nDataSet.Dimesion = %d\nDataSet.Cities: " %
                (self.name_ds, self.dimension)) + "\n[\n\t" + "\n\t".join([str(city) for city in self.cities]) + "\n]"
# </editor-fold>


def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    # bar = u"\u2588" * filledLength + ' ' * (barLength - filledLength)
    bar = u"\u2588" * filledLength + u"\u2005\u2005\u2005" * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

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
    for j in range(10):
        # New Generation
        print "======================="
        print "iteration:", j
        new_generation = []
        for i in range(0, g_POPULATION_SIZE, 2):
            printProgress(i, g_POPULATION_SIZE)
            parent1 = Dataset.tournament_selection(tour_list)
            parent2 = Dataset.tournament_selection(tour_list)
            child1, child2 = ds.crossover(parent1, parent2)
            if child1 is None or child2 is None:
                print "child[" + str(child1) + "] , child2[" + str(child2)
            new_generation.append(child1)
            new_generation.append(child2)
        printProgress(1, 1)
        # print "tour_list:"
        # for tour in tour_list:
        #     print "tour:"
        #     for city in tour:
        #         print city.name_c,
        #     print "----------"
        # print "new_tour_list:"
        # for tour in new_generation:
        #     print "tour:"
        #     for city in tour:
        #         print city.name_c,
        #     print "----------"
        tour_list = []
        tour_index = -1
        for tour in new_generation:
            tour_list.append(list())
            tour_index += 1
            for city in tour:
                if city is None:
                    break
                else:
                    tour_list[tour_index].append(city)
        # tour_list = [tour[:] for tour in new_generation]
        # </editor-fold>
