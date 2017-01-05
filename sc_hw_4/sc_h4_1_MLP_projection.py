import numpy as np
import os
import sys
import matplotlib.pyplot as plt

g_DIRECTORY_TRAIN = "Characters-TrainSetHW4/"
g_DIRECTORY_TEST = "Characters-TestSetHW4/"

g_TRAIN_ITERATION_COUNT = 1000
g_LEARNING_RATE = 0.1
g_CONVERGENCE_THRESHOLD = 0.1

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

    def __init__(self, input_size, output_size, hidden_size):
        self.__neuron_count = output_size
        self.__hidden_neurons = list()
        self.__neurons = list()
        # Initialize Hidden Layer
        default_w = np.random.randint(0, 21) / 100.
        default_w = 0.05
        Neuron.reset_id(100)
        # hidden_size = (output_size + input_size) / 2
        for i in range(hidden_size):
            neuron = Neuron(input_size, default_w)
            self.__hidden_neurons.append(neuron)
        # Initialize Output Layer
        # default_w = np.random.randint(0, 21) / 100.
        Neuron.reset_id(200)
        for i in range(output_size):
            neuron = Neuron(hidden_size, default_w)
            self.__neurons.append(neuron)

    def train(self, data, target):
        max_dw = 0

        # --- Feed Forward ---
        # Compute Hidden Layer`s Output
        output_layer_inputs = list()
        for neuron in self.__hidden_neurons:
            hidden_layer_output = neuron.get_output(data)
            output_layer_inputs.append(hidden_layer_output)
        # --- Back Propagation ---
        # Train Output Layer
        target_list = [-1] * self.__neuron_count
        target_list[target] = 1
        err_list = list()
        weights_list = list()
        output_layer_dw = dict()
        i = -1
        # Output Layer Neurons
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

            dw = np.abs(dw_list)
            dw = np.max(dw)
            if max_dw < dw:
                max_dw = dw
        # Hidden Layer Neurons
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

            dw = np.abs(dw_list)
            dw = np.max(dw)
            if max_dw < dw:
                max_dw = dw

        return max_dw

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
            # if output > Neuron.threshold:
            #     return i
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
    data = np.array(data_value_list).reshape((g_DATA_ROWS_COUNT, g_DATA_COLUMNS_COUNT))
    projected_list = [0] * (g_DATA_ROWS_COUNT + g_DATA_COLUMNS_COUNT)
    for i in range(g_DATA_ROWS_COUNT):
        for j in range(g_DATA_COLUMNS_COUNT):
            if data[i, j] == 1:
                projected_list[i] += 1
                projected_list[g_DATA_ROWS_COUNT + j] += 1
    return projected_list, data_char_map


def network_training(network, train_data_list, label_list, threshold, minimum_iteration=50):
    iter = 1
    while true:
        iter += 1
        max_dw = 0
        n = len(label_list)
        for i in range(n):
            train_data = train_data_list[i]
            label = label_list[i]
            dw = network.train(train_data, label)
            # print dw
            if max_dw < dw:
                max_dw = dw
        if (max_dw < threshold) and (iter > minimum_iteration):
            break
        if iter > MLP.training_count:
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
    MLP.training_count = g_TRAIN_ITERATION_COUNT

    mlp_12 = MLP(g_DATA_ROWS_COUNT + g_DATA_COLUMNS_COUNT, g_CHARACTER_COUNT, 12)
    data_char_map = CharMap()
    label_char_map = CharMap()

    # <editor-fold desc = "Train MLP">
    train_data_list = list()
    train_label_list = list()
    for file in os.listdir(g_DIRECTORY_TRAIN):
        if file.endswith(".txt"):
            relative_path = g_DIRECTORY_TRAIN + file
            label = label_char_map.get_mapped_value(str(file)[0])
            train_data, data_char_map = load_data(relative_path, data_char_map)
            train_data_list.append(train_data)
            train_label_list.append(label)

    mlp_12, iter_count = network_training(mlp_12, train_data_list, train_label_list, Neuron.threshold, 200)
    # </editor-fold>

    # <editor-fold desc="Predict the test data">
    test_data_list = list()
    test_label_list = list()


    prediction_error = network_prediction(mlp_12, test_data_list, test_label_list)
    print "Error Rate (12 Neuron Hidden Layer): %f" % prediction_error
    # </editor-fold>

    # <editor-fold desc="Report">
    # step_count = 100
    # # threshold_list = [(1. * i / step_count / 10.) for i in range(step_count + 1)]
    # threshold_list = [(i / 2000.) for i in range(1, step_count + 1, 5)]
    # max_theta = max(threshold_list)
    # iter_count_list = list()
    # prediction_error_list = list()
    # for threshold in threshold_list:
    #     printProgress(threshold, max_theta, "Plot Progress :", "(" + u'\u0398' + " = %0.4f)" % threshold)
    #     mlp1 = MLP(g_DATA_ROWS_COUNT * g_DATA_COLUMNS_COUNT, g_CHARACTER_COUNT, 35)
    #     mlp1, iter_count = network_training(mlp1, train_data_list, test_label_list, threshold, 5)
    #     prediction_error = network_prediction(mlp1, test_data_list, test_label_list)
    #     iter_count_list.append(iter_count)
    #     prediction_error_list.append(prediction_error)
    #
    # plt.title("Multilayer Perceptron")
    # plt.xlabel("Threshold")
    # plt.ylabel("Iteration | Error")
    # plt.text(threshold_list[0], iter_count_list[0], "Training Iteration")
    # plt.plot(threshold_list, iter_count_list, '--')
    # plt.text(threshold_list[0], prediction_error_list[0], "Prediction Error")
    # plt.plot(threshold_list, prediction_error_list, '-.')
    # plt.show()
    #
    # print
    # print "Errors :"
    # print [float("%.2f" % err) for err in prediction_error_list]
    # print "Iterations :"
    # print iter_count_list
    # </editor-fold>
