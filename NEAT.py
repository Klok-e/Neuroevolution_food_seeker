import numpy as np
import random
import math
import pygame

CHANCE_MUTATION = 0.3
CHANCE_ADD_CONNECTION = 0.3
CHANCE_REMOVE_CONNECTION = 0.3
CHANCE_ADD_NEURON = 0.2
CHANCE_REMOVE_NEURON = 0.2
CHANCE_CHANGE_WEIGHT = 0.5
CHANCE_STRUCTURAL_CHANGE = 0.3


class Connection():
    CHANCE_RANDOMIZE_WEIGHT = 0.2

    def __init__(self, from_neuron, to_neuron, weight=None):
        if weight != None:
            self.weight = weight
        else:
            self.weight = random.uniform(-2, 2)
        self.to_neuron = to_neuron
        self.from_neuron = from_neuron

        self.to_neuron.in_connections.append(self)
        self.from_neuron.out_connections.append(self)

    def mutate_weight(self):
        weight = self.weight
        if random.random() < Connection.CHANCE_RANDOMIZE_WEIGHT:
            weight = random.uniform(-2, 2)
        else:
            weight += random.uniform(-0.5, 0.5)
        return weight

    def transfer(self):
        if self.from_neuron.out != None:
            self.to_neuron.inputs.append(self.from_neuron.out * self.weight)
            self.from_neuron.transferred += 1
            if self.from_neuron.transferred == len(self.from_neuron.out_connections):
                self.from_neuron.out = None  # indicates that output has been transferred
                self.from_neuron.transferred = 0
            return True
        return False

    def terminate(self):
        self.to_neuron.in_connections.remove(self)
        self.from_neuron.out_connections.remove(self)

    def __str__(self):
        return str(round(self.weight, 4)) + ' ' + str(self.from_neuron.id) + ' ' + str(self.to_neuron.id)


class Neuron():
    bias_out = 1
    NUMBER = 0

    def __init__(self, input_neur=False, output_neur=False, bias=False):
        self.id = Neuron.NUMBER
        Neuron.NUMBER += 1

        self.is_input_neur = input_neur
        self.is_output_neur = output_neur
        self.bias = bias

        self.in_connections = []
        self.out_connections = []

        self.transferred = 0

        self.inputs = []
        self.out = None
        if self.bias: self.out = self.bias_out

    def calculate_out(self):
        if len(self.inputs) == len(self.in_connections) or self.is_input_neur:
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

    def __str__(self):
        return str(self.id) + ' ' + str(self.is_input_neur) + ' ' + str(self.is_output_neur)


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

        for i in range(len(self.input_neurons) - 1):
            neur = self.input_neurons[i]
            neur.inputs = [data[i], ]
            neur.calculate_out()

        prediction = [0 for i in range(len(self.output_neurons))]
        not_transferred = self.connections.copy()
        while len(not_transferred) != 0:
            for conn in not_transferred:
                b = conn.transfer()
                if b:
                    not_transferred.remove(conn)

        for i, out_neur in enumerate(self.output_neurons):
            out_neur.calculate_out()
            prediction[i] = out_neur.out
        return prediction

    def set_structure(self, struct):
        self.input_neurons, self.h_neurons, self.output_neurons, self.connections = struct

    def get_structure_data(self):
        input_neurons = [Neuron(input_neur=True) for neu in self.input_neurons]
        h_neurons = [Neuron() for neu in self.h_neurons]
        output_neurons = [Neuron(output_neur=True) for neu in self.output_neurons]

        connections = [0 for conn in self.connections]

        for i, conn in enumerate(self.connections):
            if conn.from_neuron.is_input_neur:
                if conn.to_neuron.is_output_neur:
                    connections[i] = Connection(input_neurons[self.input_neurons.index(conn.from_neuron)],
                                                output_neurons[self.output_neurons.index(conn.to_neuron)],
                                                conn.weight)
                else:  # to neur is in hidden layer
                    connections[i] = Connection(input_neurons[self.input_neurons.index(conn.from_neuron)],
                                                h_neurons[self.h_neurons.index(conn.to_neuron)],
                                                conn.weight)
            else:  # from neur is in hidden layer
                if conn.to_neuron.is_output_neur:
                    connections[i] = Connection(h_neurons[self.h_neurons.index(conn.from_neuron)],
                                                output_neurons[self.output_neurons.index(conn.to_neuron)],
                                                conn.weight)
                else:  # to neur is in hidden layer
                    connections[i] = Connection(h_neurons[self.h_neurons.index(conn.from_neuron)],
                                                h_neurons[self.h_neurons.index(conn.to_neuron)],
                                                conn.weight)

        return input_neurons, h_neurons, output_neurons, connections

    def create_offspring(self):
        offspring = Network(len(self.input_neurons) - 1, len(self.output_neurons))
        offspring.set_structure(self.get_structure_data())

        if random.random() < CHANCE_MUTATION:
            while True:
                if random.random() < CHANCE_CHANGE_WEIGHT:
                    if len(offspring.connections) > 0:
                        random.choice(offspring.connections).mutate_weight()
                        break
                if random.random() < CHANCE_STRUCTURAL_CHANGE:
                    if random.random() < CHANCE_ADD_CONNECTION:
                        all_neurons = offspring.input_neurons + offspring.h_neurons + offspring.output_neurons
                        while True:
                            n1, n2 = random.choices(all_neurons, k=2)
                            if n1.is_input_neur and n2.is_input_neur:
                                continue
                            elif n1.is_output_neur and n2.is_output_neur:
                                continue
                            elif n1.is_output_neur:  # output can't be the first neuron
                                continue
                            elif n2.is_input_neur:  # input can't be the second neuron
                                continue
                            else:
                                offspring.connections.append(Connection(n1, n2))
                                break
                        break
                    elif random.random() < CHANCE_REMOVE_CONNECTION:
                        if len(offspring.connections) > 0:
                            conn = random.choice(offspring.connections)
                            conn.terminate()
                            offspring.connections.remove(conn)
                            break
                    elif random.random() < CHANCE_ADD_NEURON:
                        if len(offspring.connections) > 0:
                            conn = random.choice(offspring.connections)
                            n1, n2, w = conn.from_neuron, conn.to_neuron, conn.weight
                            conn.terminate()
                            offspring.connections.remove(conn)
                            n3 = Neuron()
                            offspring.h_neurons.append(n3)
                            offspring.connections.append(Connection(n1, n3, 1))
                            offspring.connections.append(Connection(n3, n2, w))
                            break
                    elif random.random() < CHANCE_REMOVE_NEURON:
                        not_connected = []
                        for neuron in offspring.h_neurons:
                            if len(neuron.in_connections) + len(neuron.out_connections) == 0:
                                not_connected.append(neuron)
                        if len(not_connected) != 0:
                            to_remove = random.choice(not_connected)
                            offspring.h_neurons.remove(to_remove)
                            break
        return offspring

    def get_image_of_network(self):
        pass

    def __str__(self):
        data = self.get_structure_data()
        return str((list(map(str, data[0])), list(map(str, data[1])), list(map(str, data[2])), list(map(str, data[3]))))


if __name__ == '__main__':
    n = Network(2, 3)
    res = n.predict((-0.2, 0.5))
    print(res)

    #print(n)

    offspring = n.create_offspring()
    #print(offspring)
    # data = n.get_structure_data()
    # print(list(map(str, data[0])), list(map(str, data[1])), list(map(str, data[2])), list(map(str, data[3])))
