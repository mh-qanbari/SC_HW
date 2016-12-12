from random import shuffle

g_POPULATION_SIZE = 50

g_ALI535_FILENAME = "ali535.tsp"
g_SELECTED_DATASET_FILE = g_ALI535_FILENAME
g_DS_FEATURE_NAME = "NAME"
g_DS_FEATURE_DIMENSION = "DIMENSION"
g_DS_FEATURE_BEGIN_DATA = "NODE_COORD_SECTION"
g_DS_FEATURE_EOF = "EOF"
false = False
true = True

class City:
    name_c = str(None)
    coord_x = None
    coord_y = None

    def __str__(self):
        return "City[%s] : coord = (%.2f, %.2f)" % (self.name_c, self.coord_x, self.coord_y)


class DataSet:
    # Public:
    name_ds = None
    dimension = None
    cities = list()
    # Private:
    __answer__ = None

    # <editor-fold desc="Generate random population">
    def generate_population(self, population_size=g_POPULATION_SIZE):
        chromosomes = []
        for i in range(population_size):
            chromosome = range(self.dimension)
            shuffle(chromosome)
            chromosomes.append(chromosome)
        return chromosomes
    # </editor-fold>

    def get_answer(self):
        return self.__answer__

    def __str__(self):
        return ("DataSet.name = %s\nDataSet.Dimesion = %d\nDataSet.Cities: " %
                (self.name_ds, self.dimension)) + "\n[\n\t" + "\n\t".join([str(city) for city in self.cities]) + "\n]"


if __name__ == "__main__":
    ds = DataSet()

    # Load dataset
    with open(g_SELECTED_DATASET_FILE, mode='r') as ds_file:
        read_data = false
        for line in ds_file:
            line = line.rstrip()
            if line == g_DS_FEATURE_EOF:
                break
            words = line.split(':')
            # words = line.split()
            key = words[0].replace(" ", '')
            if read_data:
                city = City()
                # words = line.split(" ")
                words = line.split()
                city.name_c = words[0]
                city.coord_x = float(words[1])
                city.coord_y = float(words[2])
                ds.cities.append(city)
            elif key == g_DS_FEATURE_NAME:
                ds.name_ds = words[1].replace(" ", '')
            elif key == g_DS_FEATURE_DIMENSION:
                ds.dimension = int(words[1].replace(" ", ''))
            elif key == g_DS_FEATURE_BEGIN_DATA:
                read_data = true

    # Generate population
    chromosomes = ds.generate_population()

    # <editor-fold desc="">
    # </editor-fold>
