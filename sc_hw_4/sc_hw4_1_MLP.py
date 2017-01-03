# import random
import numpy as np
import os
import sys

g_DIRECTORY_TRAIN = "Characters-TrainSetHW4/"
g_DIRECTORY_TEST = "Characters-TestSetHW4/"

g_TRAIN_ITERATION_COUNT = 100
g_LEARNING_RATE = 0.1
g_CONVERGENCE_THRESHOLD = 0.9

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
    threshold = 0.5
    __next_id = 0

    def __init__(self, size=None, default_weight=0.2):
        self.__id = Neuron.__next_id
        Neuron.__next_id += 1
        self.weights = np.array([])
        if size is not None:
            self.init(size, default_weight)

    def init(self, size, default_weight):
        for _ in range(size + 1):
            self.weights = np.append(self.weights, [default_weight])

    def train(self, data, target):
        old_weights = np.copy(self.weights)
        data_ = np.append([1], data)
        value = sum(data_ * self.weights)
        y_in = Neuron.step_function(value)
        err = target - y_in
        self.weights += Neuron.learning_rate * err * data_
        dw_array = np.abs(self.weights - old_weights)
        max_dw = np.max(dw_array)
        return max_dw

    def get_output(self, data):
        data_ = np.append([1], data)
        value = sum(data_ * self.weights)
        return Neuron.step_function(value)

    @staticmethod
    def step_function(value):
        if value > -710.0:
            return 2. / (1. + np.exp(-value)) - 1.
        return -1

    @staticmethod
    def reset_id(start_id):
        Neuron.__next_id = start_id


class MLP:
    training_count = int(50)

    def __init__(self, input_size, output_size):
        self.__neuron_count = output_size
        self.__hidden_neurons = list()
        self.__neurons = list()
        default_w = np.random.randint(0, 21) / 100.
        # Initialize Hidden Layer
        Neuron.reset_id(100)
        hidden_size = (output_size + input_size) / 2
        for i in range(hidden_size, ):
            neuron = Neuron(input_size, default_w)
            self.__hidden_neurons.append(neuron)
        # Initialize Output Layer
        Neuron.reset_id(200)
        for i in range(output_size):
            neuron = Neuron(hidden_size, default_w)
            self.__neurons.append(neuron)

    def train(self, data, target):
        # --- Feed Forward ---
        # Compute Hidden Layer`s Output
        output_layer_inputs = list()
        for neuron in self.__hidden_neurons:
            hidden_layer_output = neuron.get_output(data)
            output_layer_inputs.append(hidden_layer_output)
        # --- Back Propagation ---
        # Train Output Layer
        # output_layer_outputs = list()
        target_list = [-1] * self.__neuron_count
        target_list[target] = 1
        err_list = list()
        weights_list = list()
        output_layer_dw = dict()
        i = -1
        for neuron in self.__neurons:
            i += 1
            output = neuron.get_output(output_layer_inputs)
            weights = np.copy(neuron.weights)
            weights_list.append(weights)
            err = (target_list[i] - output) * 0.5 * (1 + output) * (1 - output)
            dw_list = Neuron.learning_rate * err * np.append([1], output_layer_inputs)
            output_layer_dw[neuron] = dw_list
            err_list.append(err)
            neuron.weights = np.copy(weights + dw_list)
            self.__neurons[i] = neuron
        for i in range(len(self.__hidden_neurons)):
            neuron = self.__hidden_neurons[i]
            old_weights = np.copy(neuron.weights)
            err_in = sum(np.array(err_list) * np.array(weights_list)[:, i])
            neuron_output = output_layer_inputs[i]
            err = err_in * 0.5 * (1 + neuron_output) * (1 - neuron_output)
            dw_list = Neuron.learning_rate * err * np.append([1], data)
            old_weights += dw_list
            neuron.weights = np.copy(old_weights)
            self.__hidden_neurons[i] = neuron

    def predict(self, data):
        out_lay_in_list = list()
        for i in range(len(self.__hidden_neurons)):
            neuron = self.__hidden_neurons[i]
            output = neuron.get_output(data)
            out_lay_in_list.append(output)
        max_i = -1
        max_out = -2
        i = -1
        for neuron in self.__neurons:
            i += 1
            output = neuron.get_output(out_lay_in_list)
            if max_out < output:
                max_out = output
                max_i = i
            if output > Neuron.threshold:
            # if output == 1:
                print output
                return i
        return max_i


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
    MLP.training_count = g_TRAIN_ITERATION_COUNT
    mlp = MLP(g_DATA_ROWS_COUNT * g_DATA_COLUMNS_COUNT, g_CHARACTER_COUNT)
    data_char_map = CharMap()
    label_char_map = CharMap()

    # <editor-fold desc = "Train MLP">
    for i in range(g_TRAIN_ITERATION_COUNT):
        printProgress(i, g_TRAIN_ITERATION_COUNT, "Network Training:")
        for file in os.listdir(g_DIRECTORY_TRAIN):
            if file.endswith(".txt"):
                relative_path = g_DIRECTORY_TRAIN + file
                label = label_char_map.get_mapped_value(str(file)[0])
                train_data, data_char_map = load_data(relative_path, data_char_map)
                train_data = train_data
                mlp.train(train_data, label)
    printProgress(1, 1, "Network Training:")
    print
    # </editor-fold>

    # <editor-fold desc="Predict the test data">
    predicted_status_list = list()
    for file in os.listdir(g_DIRECTORY_TEST):
        if file.endswith(".txt"):
            relative_path = g_DIRECTORY_TEST + file
            label = label_char_map.get_mapped_value(str(file)[0])
            test_data, data_char_map = load_data(relative_path, data_char_map)
            test_data = test_data
            predicted_value = mlp.predict(test_data)
            predicted = label_char_map.get_mapped_char(predicted_value)
            predicted_status = (predicted_value == label)
            predicted_status_list.append(predicted_status)
            print file
            print predicted_value, "==", label
            print predicted_status, predicted
            print

    n = len(predicted_status_list)
    true_status_count = np.count_nonzero(predicted_status_list)
    n_err = n - true_status_count
    print "Error Rate: %f" % ((float(n_err) / n) * 100)
    # </editor-fold>
