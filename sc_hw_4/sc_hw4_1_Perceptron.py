# import random
import numpy as np
import os

g_DIRECTORY_TRAIN = "Characters-TrainSetHW4/"
g_DIRECTORY_TEST = "Characters-TestSetHW4/"

g_TRAIN_ITERATION_COUNT = 10
g_LEARNING_RATE = 0.5

g_DATA_ROWS_COUNT = 9
g_DATA_COLUMNS_COUNT = 7
g_CHARACTER_COUNT = 7

true = True
false = False


class CharMap:
    __static_char_map = dict()
    __static_new_value = 0

    def __init__(self):
        self.__char_map = dict()
        self.__new_value = 0

    def get_mapped_value(self, obj, offered=None):
        if obj in self.__char_map:
            return self.__char_map[obj]
        else:
            value = self.__new_value
            self.__new_value += 1
            self.__char_map[obj] = value
            return value
            # if offered is None:
            #     self.__new_value += 1
            #     self.__char_map[obj] = value
            #     return value
            # else:
            #     self.__char_map[obj] = offered
            #     if self.__new_value < offered:
            #         self.__new_value = offered + 1
            #     return offered

    def get_mapped_char(self, value):
        for obj, val in self.__char_map.iteritems():
            if val == value:
                return obj
        return None

    @staticmethod
    def compute_mapped_value(obj):
        if obj in CharMap.__static_char_map:
            return CharMap.__static_char_map[obj]
        else:
            value = CharMap.__static_new_value
            CharMap.__static_new_value += 1
            CharMap.__static_char_map[obj] = value
            return value

    @staticmethod
    def copute_mapped_char(value):
        for obj, val in CharMap.__static_char_map.iteritems():
            if val == value:
                return obj
        return None


class Neuron:
    learning_rate = float(1)
    threshold = 1

    def __init__(self, size=None):
        self.weights = np.array([])
        if size is not None:
            self.init(size)

    def init(self, size):
        for i in range(size + 1):
            self.weights = np.append(self.weights, 0)

    def train(self, data, target):
        data_ = np.append([1], data)
        i = -1
        while true:
            i += 1
            # old_weights = np.copy(self.weights)
            value = sum(data_ * self.weights)
            y_in = Neuron.__activation_function(value)
            max_dw = 0
            if y_in != target:
                dw = Neuron.learning_rate * target * data_
                self.weights += dw
                max_dw = max(dw)
            return max_dw
            # convergence_status = np.equal(old_weights, self.weights)
            # if np.all(convergence_status):
            #     break

    def get_output(self, data):
        data_ = np.append([1], data)
        value = sum(data_ * self.weights)
        return Neuron.__activation_function(value)

    @staticmethod
    def __activation_function(value):
        if value > Neuron.threshold:
            return 1
        elif value < -Neuron.threshold:
            return -1
        else:
            return 0


class Perceptron:
    training_count = int(50)

    def __init__(self, input_size, output_size):
        self.__neuron_count = output_size
        self.__neurons = list()
        for _ in range(output_size):
            neuron = Neuron(input_size)
            self.__neurons.append(neuron)

    def train(self, data, target):
        target_list = [-1] * self.__neuron_count
        target_list[target] = 1
        max_dw = 0
        for i in range(self.__neuron_count):
            neuron = self.__neurons[i]
            dw = neuron.train(data, target_list[i])
            self.__neurons[i] = neuron
            if max_dw < dw:
                max_dw = dw
        return max_dw

    def predict(self, data):
        for i in range(self.__neuron_count):
            neuron = self.__neurons[i]
            if neuron.get_output(data) == 1:
                return i


def load_data(file_address, data_char_map):
    data_value_list = list()
    with open(file_address, mode='r') as file_content:
        for line in file_content:
            line = line.replace('\n', '')
            for char in line:
                value = data_char_map.get_mapped_value(char)
                data_value_list.append(value)
    return data_value_list, data_char_map

if __name__ == "__main__":
    perceptron = Perceptron(g_DATA_ROWS_COUNT * g_DATA_COLUMNS_COUNT, g_CHARACTER_COUNT)
    data_char_map = CharMap()
    label_char_map = CharMap()

    # <editor-fold desc = "Train perceptron">
    for i in range(Perceptron.training_count):
        for file in os.listdir(g_DIRECTORY_TRAIN):
            if file.endswith(".txt"):
                relative_path = g_DIRECTORY_TRAIN + file
                label = label_char_map.get_mapped_value(str(file)[0])
                train_data, data_char_map = load_data(relative_path, data_char_map)
                train_data = train_data
                perceptron.train(train_data, label)
                # print file
                # print p.weights
                # print
    # </editor-fold>

    # <editor-fold desc="Predict the test data">
    predicted_status_list = list()
    for file in os.listdir(g_DIRECTORY_TEST):
        if file.endswith(".txt"):
            relative_path = g_DIRECTORY_TEST + file
            label = label_char_map.get_mapped_value(str(file)[0])
            test_data, data_char_map = load_data(relative_path, data_char_map)
            test_data = test_data
            # p = Perceptron(len(train_data))
            predicted_value = perceptron.predict(test_data)
            predicted = label_char_map.get_mapped_char(predicted_value)
            predicted_status = (predicted_value == label)
            predicted_status_list.append(predicted_status)
            print file
            print predicted_status
            print

    n = len(predicted_status_list)
    true_status_count = np.count_nonzero(predicted_status_list)
    n_err = n - true_status_count
    print "Error Rate: %f" % ((float(n_err) / n) * 100)
    # </editor-fold>
