import numpy as np
import random
import neat
import math


class Connection():
    def __init__(self, from_neuron, to_neuron):
        self.weight = random.uniform(-2, 2)
        self.to_neuron = to_neuron
        self.from_neuron = from_neuron

    def transfer(self, inp):
        self.to_neuron.inputs.append(inp * self.weight)


class Neuron():
    bias_out = 1

    def __init__(self, input_neur=False, output_neur=False, bias=False):
        self.is_input_neur = input_neur
        self.is_output_neur = output_neur
        self.bias = bias

        self.inputs = []
        self.out = None

        if self.bias: self.out = self.bias_out

    def calculate_out(self):
        su = 0
        for numb in self.inputs:
            su += numb
        if not self.is_input_neur:  # if neuron is not input
            self.out = self.activate(su)
        else:
            self.out = su
        self.inputs.clear()

    def activate(self, x):
        a = math.tanh(x)
        return a


class Network():
    def __init__(self, input_neurons, output_neurons):
        self.input_neurons = [Neuron(input_neur=True) for i in range(input_neurons)]
        self.input_neurons.append(Neuron(bias=True))  # bias
        self.output_neurons = [Neuron(output_neur=True) for i in range(output_neurons)]

        self.h_neurons = []
        self.connections = []

    def predict(self, data):
        if len(data) != len(self.input_neurons) - 1:
            raise ValueError('Wrong data length')

        for i, neur in enumerate(self.input_neurons):
            neur.inputs = [data[i], ]


if __name__ == '__main__':
    pass
