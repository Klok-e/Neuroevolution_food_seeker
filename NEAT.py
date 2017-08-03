import numpy as np
import random
import math
import pygame


class Connection():
    def __init__(self, from_neuron, to_neuron):
        self.weight = random.uniform(-2, 2)
        self.to_neuron = to_neuron
        self.from_neuron = from_neuron

        self.to_neuron.in_connections.append(self)
        self.from_neuron.out_connections.append(self)

    def transfer(self):
        if self.from_neuron.out != None:
            self.to_neuron.inputs.append(self.from_neuron.out * self.weight)
            self.from_neuron.out = None  # indicates that output has been transferred
            return True
        return False


class Neuron():
    bias_out = 1

    def __init__(self, input_neur=False, output_neur=False, bias=False):
        self.is_input_neur = input_neur
        self.is_output_neur = output_neur
        self.bias = bias

        self.in_connections = []
        self.out_connections = []

        self.inputs = []
        self.out = None

    def calculate_out(self):
        if len(self.inputs) == len(self.in_connections):
            su = 0
            for numb in self.inputs:
                su += numb
            if not self.is_input_neur:  # if neuron is not input
                self.out = self.activate(su)
            elif self.bias:
                self.out = self.bias_out
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
            neur.calculate_out()

        prediction = [0 for i in range(len(self.output_neurons))]
        not_transferred = self.connections.copy()
        while len(not_transferred) != 0:
            for conn in not_transferred:
                b = conn.transfer()
                if b:
                    not_transferred.remove(conn)

        for i,out_neur in enumerate(self.output_neurons):
            prediction[i]=out_neur.out
        return prediction


if __name__ == '__main__':
    pass
