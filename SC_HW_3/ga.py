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

# <editor-fold desc="Necessary Standards">
from time import gmtime, strftime

g_OUTPUT_FILE = strftime("output_%Y%m%d%H%M%S", gmtime())
file_output = open(g_OUTPUT_FILE, 'w')


def printl(obj):
    orig_stdout = sys.stdout
    sys.stdout = file_output
    print obj
    sys.stdout = orig_stdout
    print obj


def print_(obj):
    orig_stdout = sys.stdout
    sys.stdout = file_output
    print obj,
    sys.stdout = orig_stdout
    print obj,
# </editor-fold>

# <editor-fold desc="Parameters">
g_POPULATION_SIZE = 50
g_CROSSOVER_PROBABILITY = 0.9
g_MUTATION_PROBABILITY = 0.02
g_REPLACEMENT_NEW_GENERATION_POSSIBILITY = 0.80

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
            # self.suggest(tour)
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
            printl('test')
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
        if (self.__best_distance is None) or (len(self.__best_tour) == 0) or (dist < self.__best_distance):
            self.__best_tour = tour
            self.__best_distance = dist
        return dist

    def reset(self):
        self.__best_distance = None
        self.__best_tour = list()

    def reset(self, tour):
        self.__best_distance = Dataset.compute_distance()
        self.__best_tour = tour

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
        if len(parent1) != 535 or len(parent2) != 535:
            printl('here')
        global child1
        global child2
        child1 = [None for _ in range(len(parent1))]
        child2 = [None for _ in range(len(parent2))]
        crossover_point1 = random.randint(0, self.dimension - 1)
        crossover_point2 = random.randint(0, self.dimension - 1)
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

    # <editor-fold desc="Mutation">
    @staticmethod
    def insertion_mutate(tour):
        tour_size = len(tour)
        index1 = random.randint(0, tour_size - 1)
        index2 = random.randint(0, tour_size - 1)
        if index1 > index2:
            temp = index1
            index1 = index2
            index2 = temp
        city = tour[index2]
        del tour[index2]
        tour.insert(index1, city)
        return tour
    # </editor-fold>

    def get_best_tour(self):
        return self.__best_tour

    def __str__(self):
        return ("DataSet.name = %s\nDataSet.Dimesion = %d\nDataSet.Cities: " %
                (self.name_ds, self.dimension)) + "\n[\n\t" + "\n\t".join([str(city) for city in self.cities]) + "\n]"
# </editor-fold>


def printProgress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    format_ = "{0:." + str(decimals) + "f}"
    percent = format_.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    # bar = u"\u2588" * filledLength + ' ' * (barLength - filledLength)
    bar = u"\u2588" * filled_length + u"\u2005\u2005\u2005" * (bar_length - filled_length)
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

    for j in range(100):
        printl("=======================")
        printl("iteration:" + str(j))
        new_generation = []
        # <editor-fold desc="Crossover">
        crossover_random = random.random()
        if crossover_random < g_CROSSOVER_PROBABILITY:
            # New Generation
            for i in range(0, g_POPULATION_SIZE, 2):
                printProgress(i + 2, g_POPULATION_SIZE, "Crossover:")
                parent1 = Dataset.tournament_selection(tour_list)
                parent2 = Dataset.tournament_selection(tour_list)
                child1, child2 = ds.crossover(parent1, parent2)
                if child1 is None or child2 is None:
                    printl("child[" + str(child1) + "] , child2[" + str(child2))
                new_generation.append(child1)
                new_generation.append(child2)
                # tour_list = []
                # tour_index = -1
                # for tour in new_generation:
                #     tour_list.append(list())
                #     tour_index += 1
                #     for city in tour:
                #         if city is None:
                #             break
                #         else:
                #             tour_list[tour_index].append(city)
                # tour_list = [tour[:] for tour in new_generation]
        else:
            new_generation = [tour[:] for tour in tour_list]
            printProgress(g_POPULATION_SIZE, g_POPULATION_SIZE, "Crossover:")
        # </editor-fold>
        print
        # <editor-fold desc="Mutation">
        for tour_index in range(g_POPULATION_SIZE):
            printProgress(tour_index+1, g_POPULATION_SIZE, "Mutation: ")
            mutation_random = random.random()
            if mutation_random < g_MUTATION_PROBABILITY:
                tour = new_generation[tour_index]
                new_generation[tour_index] = Dataset.insertion_mutate(tour)
        # </editor-fold>

        # <editor-fold desc="Replacement">
        # tour_list = []
        tour_index = -1
        for tour in new_generation:
            tour_index += 1
            printProgress(tour_index+1, g_POPULATION_SIZE, "Replacement:")
            replacement_rand = random.random()
            if replacement_rand < g_REPLACEMENT_NEW_GENERATION_POSSIBILITY:
                # tour_list.append(tour[:])
                tour_list[tour_index] = tour[:]
        # </editor-fold>

        # <editor-fold desc="Show the best answer found at this iteration">
        min_d = sys.maxint
        index = -1
        for tour in new_generation:
            dist = Dataset.compute_distance(tour)
            if min_d > dist:
                min_d = dist
                ds.suggest(tour)
        printl("Answer :" + str(min_d))
        # </editor-fold>

    if ds.get_best_tour() is not None:
        printl("Best answer :" + str(ds.get_distance()))
