"""
# Author : Mohammadhossein Qanbari
# St. No. : 830595021
# Ant Colony System for Travelling Salesman Problem
# Dataset: ali535, bayg29, d2103
"""

from math import sin, cos, asin, sqrt
import random
import sys

# <editor-fold desc="ACS Parameters">
g_ANT_COUNT = 535
g_CITY_COUNT = 20
g_ITERATION_COUNT = 10
g_ALI535_FILE = "ali535.tsp"
g_BAYG29_FILE = "bayg29.tsp"
g_d2103_FILE = "d2103.tsp"
g_SELECTED_FILE = g_ALI535_FILE

true = True
false = False
# </editor-fold>


def insertion_sort(alist, blist, clist=None):
    for index in range(1,len(alist)):
        current_value = alist[index]
        current_index = blist[index]
        if clist is not None:
            current_val2 = clist[index]
        position = index

        while (position > 0) and (alist[position-1] > current_value):
            alist[position] = alist[position-1]
            blist[position] = blist[position-1]
            if clist is not None:
                clist[position] = clist[position-1]
            position -= 1

        alist[position] = current_value
        blist[position] = current_index
        if clist is not None:
            clist[position] = current_val2


class GEO_Methods:
    EarthRadius = 6371  # kilometer

    @staticmethod
    def compute_distance(lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2.) ** 2. + cos(lat1) * cos(lat2) * sin(dlon / 2.) ** 2.
        c = 2. * asin(sqrt(a))
        return c * GEO_Methods.EarthRadius


class City:
    def __init__(self, world_size):
        # Public:
        self.index = None
        # Private:
        self.__distances = list([None for _ in range(world_size-1)])
        self.__coord_lat = None
        self.__coord_lon = None
        self.__coord_x = None
        self.__coord_y = None

    def __str__(self):
        return "City_%d" % int(self.index)

    def __eq__(self, other):
        return self.index == other.index

    def set_distance(self, another, dist):
        if another.index < self.index:
            self.__distances[another.index] = dist
        elif another.index > self.index:
            self.__distances[another.index - 1] = dist

    def get_distance(self, another):
        if another.index < self.index:
            dist = self.__distances[another.index]
            if dist is None:
                dist = another.get_distance(self)
            return dist
        elif another.index > self.index:
            dist = self.__distances[another.index - 1]
            if dist is None:
                dist = another.get_distance(self)
            return dist
        else:
            return 0

    def get_distances(self):
        return self.__distances[:]

    def set_coord_geo(self, lat, lon):
        self.__coord_lat = lat
        self.__coord_lon = lon

    def get_coord_geo(self):
        return self.__coord_lat, self.__coord_lon

    def set_coord_2d(self, x, y):
        self.__coord_x = x
        self.__coord_y = y

    def get_coord_2d(self):
        return self.__coord_x, self.__coord_y

    def set_neighbor(self, neighbor):
        if neighbor.index < self.index:
            nbr_lat, nbr_lon = neighbor.get_coord_geo()
            dist = GEO_Methods.compute_distance(self.__coord_lat, self.__coord_lon, nbr_lat, nbr_lon)
            self.__distances[neighbor.index] = dist
        elif neighbor.index > self.index:
            nbr_lat, nbr_lon = neighbor.get_coord_geo()
            dist = GEO_Methods.compute_distance(self.__coord_lat, self.__coord_lon, nbr_lat, nbr_lon)
            self.__distances[neighbor.index - 1] = dist


class Dataset:
    def __init__(self):
        # Public:
        self.name = None
        # Private:
        self.__dimension = None
        self.__cities = list()
        self.__pheromones = list()

    def init_cities(self, size):
        self.__dimension = size
        for _ in range(size):
            self.__cities.append(None)
        self.__pheromones = [[ACS.initial_pheromone for __ in range(size - 1)] for _ in range(size)]

    def get_cities(self):
        return self.__cities[:]

    def set_pheromone(self, source, destination, pheromone):
        i = source.index
        j = destination.index
        if j > i:
            j -= 1
        self.__pheromones[i][j] = pheromone
        i = destination.index
        j = source.index
        if i > j:
            i -= 1
        self.__pheromones[i][j] = pheromone

    def get_pheromone(self, source, destination):
        dest_index = destination.index
        if dest_index > source.index:
            dest_index -= 1
        return self.__pheromones[source.index][dest_index]

    def get_pheromones(self, source, destinations=None):
        if destinations is None:
            return self.__pheromones[source.index][:]
        else:
            phs = []
            for city in self.__cities:
                if city in destinations:
                    phs.append(self.__pheromones[source.index][city.index])
            return phs

    def get_dimension(self):
        if self.__dimension is None:
            return 0
        return self.__dimension

    def add_city(self, city):
        self.__cities[city.index] = city
        n = len(self.__cities)
        for i in range(n):
            c = self.__cities[i]
            if (i != city.index) and (c is not None):
                c.set_neighbor(city)
                self.__cities[i] = c

    def get_city(self, city_index):
        for city in self.__cities:
            if city.index == city_index:
                return city
        return None


class Ant:
    def __init__(self, init_city, ant_index):
        # Public:
        self.index = ant_index
        # Private:
        self.__current_city = init_city
        self.__history = list()
        self.__distance = None

    def get_current_city(self):
        return self.__current_city

    def go_next(self, next_city):
        dist = self.__current_city.get_distance(next_city)
        self.__history.append(self.__current_city)
        self.__current_city = next_city
        if self.__distance is None:
            self.__distance = dist
        else:
            self.__distance += dist

    def is_passed(self, city):
        return city in self.__history

    def get_distance(self):
        return self.__distance

    def get_forward_cities(self, cities):
        next_cities = [city for city in cities if (not (city in self.__history)) and (city != self.__current_city)]
        # n = len(cities)
        # for i in range(n):
        #     index = n - i - 1
        #     f_city = cities[index]
        #     if f_city in self.__history or city == self.__current_city:
        #         cities.pop(index)
        return next_cities


class ACS:
    # Static Class Fields
    q0 = 0.9
    initial_pheromone = 0.1
    global_evaporation = 0.4
    heuristic_power = 5
    pheromone_power = 1
    local_evaporation = 0.1

    def __init__(self, iteration_count, ant_count, dataset):
        # Public:
        self.world = dataset
        # Private:
        self.__iteration_count = iteration_count
        self.__init_pheromone = 0.1
        self.__ants = []
        for ant_index in range(ant_count):
            city_index = random.randint(0, self.world.get_dimension() - 1)
            city = self.world.get_city(city_index)
            ant = Ant(city, ant_index)
            self.__ants.append(ant)

    def start(self):
        best_distance = sys.maxint
        for i in range(self.__iteration_count):
            for ant in self.__ants:
                print "ant_%d:" % ant.index,
                curr_city = ant.get_current_city()
                cities = ant.get_forward_cities(self.world.get_cities())

                while len(cities) != 0:
                    q = random.random()
                    if q <= ACS.q0:
                        sel_params = []
                        dists = []
                        for city in cities:
                            dist = curr_city.get_distance(city)
                            if dist is None:
                                dist = city.get_distance(curr_city)
                            ph = self.world.get_pheromone(curr_city, city)
                            if dist == 0:
                                dist += 0.0001
                                city.set_distance(curr_city, dist)
                                curr_city.set_distance(city, dist)
                            sel_param = (ph ** ACS.pheromone_power) * ((1. / dist) ** ACS.heuristic_power)
                            sel_params.append(sel_param)
                            dists.append(dist)

                        insertion_sort(sel_params, cities)
                        next_city = cities.pop(0)

                        local_d_ph = 1. / (self.world.get_dimension() * ACS.initial_pheromone)
                        ph = self.world.get_pheromone(curr_city, next_city)
                        ph = (1 - ACS.local_evaporation) * ph + ACS.local_evaporation * local_d_ph

                        ant.go_next(next_city)

                    else:
                        ps = []
                        for city in cities:
                            dist = curr_city.get_distance(city)
                            if dist == 0:
                                dist += 0.0001
                                city.set_distance(curr_city, dist)
                                curr_city.set_distance(city, dist)
                            ph = self.world.get_pheromone(curr_city, city)
                            sel_param = (ph ** ACS.pheromone_power) * ((1. / dist) ** ACS.heuristic_power)
                            ps.append(sel_param)
                        ps_sum = sum(ps)
                        ps = [p / ps_sum for p in ps]
                        insertion_sort(ps, cities)

                        rand = random.random()
                        poss = 0
                        for index in range(len(ps)):
                            poss += ps[index]

                            if rand <= poss:
                                next_city = cities.pop(index)

                                local_d_ph = 1. / (self.world.get_dimension() * ACS.initial_pheromone)
                                ph = self.world.get_pheromone(curr_city, next_city)
                                ph = (1 - ACS.local_evaporation) * ph + ACS.local_evaporation * local_d_ph

                                ant.go_next(next_city)
                                break

                    cities = ant.get_forward_cities(cities)
                print ant.get_distance()

            ant_count = len(self.__ants)
            best_ant = self.__ants[0]
            for index in range(1, ant_count):
                curr_ant = self.__ants[index]
                if best_ant.get_distance() > curr_ant.get_distance():
                    best_ant = curr_ant
            d_ph = 1. / best_ant.get_distance()
            for s in range(self.world.get_dimension()):
                source = self.world.get_city(s)
                for d in range(s+1, self.world.get_dimension()):
                    destination = self.world.get_city(d)
                    sel_param = self.world.get_pheromone(source, destination)
                    sel_param = (1.0 - ACS.global_evaporation) * sel_param + d_ph
                    self.world.set_pheromone(source, destination, sel_param)
            b_dist = best_ant.get_distance()
            if best_distance > b_dist:
                best_distance = b_dist
            print "iter_%d: %f" % (i, b_dist)
        print "best_distance: ", best_distance

    # def get_best_tour(self):
    #     world_cities = self.world.get_cities()
    #     n = len(world_cities)
    #     cities = [world_cities[0]]
    #     s_city = cities[0]
    #     dist = 0
    #     for i in range(1, n):
    #         d_city = world_cities[i]


if __name__ == "__main__":
    ds = Dataset()
    with open(g_SELECTED_FILE, 'r') as file:
        read_coord = false
        for row in file:
            row = row.rstrip()
            if row == "EOF":
                break
            if read_coord:
                info = row.split()
                city = City(ds.get_dimension())
                city.index = int(info[0]) - 1
                city.set_coord_geo(float(info[1]), float(info[2]))
                ds.add_city(city)
            else:
                key = row.split(': ')[0]
                if key == "NAME":
                    ds.name = row.split(': ')[1]
                elif key == "DIMENSION":
                    ds.init_cities(int(row.split(': ')[1]))
                elif key == "NODE_COORD_SECTION":
                    read_coord = true

    acs = ACS(g_ITERATION_COUNT, g_ANT_COUNT, ds)
    acs.start()
