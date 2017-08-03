import numpy as np
import random
import neat
import math


class Connection():
    def __init__(self, neuron1, neuron2):
        self.weight = random.uniform(-2, 2)
        self.neuron1 = neuron1
        self.neuron2 = neuron2

    def transfer(self):
        self.neuron1.inp


class Neuron():
    def __init__(self, input_neur=False, output_neur=False,bias=False):
        self.is_input_neur = input_neur
        self.is_output_neur = output_neur
        if bias:
            self.inp = 1  # bias

    def activate(self, x):
        a = math.tanh(x)
        return a


class Network():
    def __init__(self, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.neurons = []
        self.connections = []


if __name__ == '__main__':
    pass
