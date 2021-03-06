# import random
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

g_DIRECTORY_TRAIN = "Characters-TrainSetHW4/"
g_DIRECTORY_TEST = "Characters-TestSetHW4/"

g_TRAIN_ITERATION_COUNT = 100
g_LEARNING_RATE = 0.4
g_CONVERGENCE_THRESHOLD = 0.01

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
            if offered is None:
                self.__new_value += 1
                self.__char_map[obj] = value
                return value
            else:
                self.__char_map[obj] = offered
                if self.__new_value < offered:
                    self.__new_value = offered + 1
                return offered

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
            self.weights = np.append(self.weights, [0])

    def train(self, data, target):
        data_ = np.append([1], data)
        value = sum(data_ * self.weights)
        y_in = Neuron.__activation_function(value)
        err = target - y_in
        dw = Neuron.learning_rate * err * data_
        self.weights += dw
        dw = np.abs(dw)
        return np.max(dw)

    def get_output(self, data):
        data_ = np.append([1], data)
        value = sum(data_ * self.weights)
        return Neuron.__activation_function(value)

    @staticmethod
    def __activation_function(value):
        theta = 1 - Neuron.threshold
        if value > theta:
            return 1
        elif value < -theta:
            return -1
        else:
            return 0


class Adaline:
    training_count = int(50)

    def __init__(self, input_size, output_size):
        self.__neuron_count = output_size
        self.__neurons = list()
        for i in range(output_size):
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
                if char == '#':
                    value = data_char_map.get_mapped_value(char, -1)
                elif char == '.':
                    value = data_char_map.get_mapped_value(char, 1)
                else:
                    value = data_char_map.get_mapped_value(char, 0)
                data_value_list.append(value)
    return data_value_list, data_char_map


def network_training(network, train_data_list, label_list, threshold):
    iter = 1
    while true:
        iter += 1
        max_dw = 0
        n = len(label_list)
        for i in range(n):
            train_data = train_data_list[i]
            label = label_list[i]
            dw = network.train(train_data, label)
            if max_dw < dw:
                max_dw = dw
        if max_dw < threshold:
            break
        if iter > Adaline.training_count:
            break
    return network, iter - 1


def network_prediction(network, test_data_list, label_list):
    incorrect_prediction = 0
    n = len(label_list)
    for i in range(n):
        test_data = test_data_list[i]
        label = label_list[i]
        predicted_value = network.predict(test_data)
        if label != predicted_value:
            incorrect_prediction += 1
    prediction_error = 100. * incorrect_prediction / n
    return prediction_error


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
    Neuron.learning_rate = g_LEARNING_RATE
    Neuron.threshold = g_CONVERGENCE_THRESHOLD
    Adaline.training_count = g_TRAIN_ITERATION_COUNT

    adaline = Adaline(g_DATA_ROWS_COUNT * g_DATA_COLUMNS_COUNT, g_CHARACTER_COUNT)
    data_char_map = CharMap()
    label_char_map = CharMap()

    # <editor-fold desc = "Train ADALIN">
    train_data_list = list()
    train_label_list = list()
    for file in os.listdir(g_DIRECTORY_TRAIN):
        if file.endswith(".txt"):
            relative_path = g_DIRECTORY_TRAIN + file
            label = label_char_map.get_mapped_value(str(file)[0])
            train_data, data_char_map = load_data(relative_path, data_char_map)
            train_data_list.append(train_data)
            train_label_list.append(label)

    adaline, iter_count = network_training(adaline, train_data_list, train_label_list, Neuron.threshold)
    # </editor-fold>

    # <editor-fold desc="Predict the test data">
    test_data_list = list()
    test_label_list = list()
    for file in os.listdir(g_DIRECTORY_TEST):
        if file.endswith(".txt"):
            relative_path = g_DIRECTORY_TEST + file
            label = label_char_map.get_mapped_value(str(file)[0])
            test_data, data_char_map = load_data(relative_path, data_char_map)
            test_data_list.append(test_data)
            test_label_list.append(label)

    prediction_error = network_prediction(adaline, test_data_list, test_label_list)
    print "Error Rate: %f" % prediction_error
    # </editor-fold>

    # <editor-fold desc="Report">
    step_count = 20
    threshold_list = [0] + [(1. * i / step_count) for i in range(1, step_count+1, 1)]
    iter_count_list = list()
    prediction_error_list = list()
    for threshold in threshold_list:
        printProgress(threshold, 1, "Plot Progress :")
        adaline1 = Adaline(g_DATA_ROWS_COUNT * g_DATA_COLUMNS_COUNT, g_CHARACTER_COUNT)
        adaline1, iter_count = network_training(adaline1, train_data_list, test_label_list, threshold)
        prediction_error = network_prediction(adaline1, test_data_list, test_label_list)
        iter_count_list.append(iter_count)
        prediction_error_list.append(prediction_error)

    plt.title("ADALINE")
    plt.xlabel("Threshold")
    plt.ylabel("Iteration | Error")
    plt.text(threshold_list[9 * step_count / 10], iter_count_list[9 * step_count / 10], "Training Iteration")
    plt.plot(threshold_list, iter_count_list, '--')
    plt.text(threshold_list[9 * step_count / 10], prediction_error_list[9 * step_count / 10], "Prediction Error")
    plt.plot(threshold_list, prediction_error_list, '-.')
    plt.axis([0, 1.0, 0, 100])
    plt.show()

    print
    # print "Thresholds :"
    # print threshold_list
    print "Errors :"
    print [float("%.2f" % err) for err in prediction_error_list]
    print "Iterations :"
    print iter_count_list
    # </editor-fold>
