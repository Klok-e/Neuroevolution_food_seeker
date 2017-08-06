import numpy as np
import random
import math
import pygame

from config import *


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
        # if self.bias: self.out = Neuron.bias_out

    def calculate_out(self):
        if self.bias:
            self.out = Neuron.bias_out
        if len(self.inputs) == len(self.in_connections) or self.is_input_neur:
            su = 0
            for numb in self.inputs:
                su += numb
            if self.is_output_neur:  # if neuron is not input (input has no activation)
                self.out = self.activate_tanh(su)
            if not self.is_input_neur and not self.is_output_neur and not self.bias:  # if is in hidden layer
                self.out = self.activate_relu(su)
            else:
                self.out = su

            self.inputs.clear()

    def activate_tanh(self, x):
        a = math.tanh(x)
        return a

    def activate_relu(self, x):
        a = max(0., x)
        return a

    def activate_sin(self, x):
        a = math.sin(x)
        return a

    def activate_linear(self, x):
        a = x
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
        # make all input neurons connected to output
        for inp in self.input_neurons:
            for out in self.output_neurons:
                self.connections.append(Connection(inp, out))

    def predict(self, data):
        if len(data) != len(self.input_neurons) - 1:
            raise ValueError('Wrong data length')

        for i in range(len(self.input_neurons) - 1):
            neur = self.input_neurons[i]
            neur.inputs = [data[i], ]
        for neur in self.input_neurons:
            neur.calculate_out()

        prediction = [0 for i in range(len(self.output_neurons))]
        not_transferred = self.connections.copy()
        while len(not_transferred) != 0:
            for neur in self.h_neurons:
                neur.calculate_out()
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

    def get_structure(self):
        struct = self.input_neurons, self.h_neurons, self.output_neurons, self.connections
        return struct

    def get_copy_structure_data(self):
        input_neurons = [Neuron(input_neur=neu.is_input_neur, bias=neu.bias) for neu in self.input_neurons]
        h_neurons = [Neuron() for neu in self.h_neurons]
        output_neurons = [Neuron(output_neur=True) for neu in self.output_neurons]

        connections = [0 for conn in self.connections]

        for i, conn in enumerate(self.connections):
            if conn.from_neuron.is_input_neur or conn.from_neuron.bias:
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
        offspring.set_structure(self.get_copy_structure_data())

        if random.random() < CHANCE_MUTATION:
            if random.random() < CHANCE_CHANGE_WEIGHT:
                if len(offspring.connections) > 0:
                    random.choice(offspring.connections).mutate_weight()

            if random.random() < CHANCE_STRUCTURAL_CHANGE:
                if random.random() < CHANCE_ADD_CONNECTION:
                    all_neurons = offspring.input_neurons + offspring.h_neurons + offspring.output_neurons
                    n1, n2 = random.sample(all_neurons, 2)
                    if n1.is_input_neur and n2.is_input_neur:
                        pass
                    elif n1.is_output_neur and n2.is_output_neur:
                        pass
                    elif n1.is_output_neur:  # output can't be the first neuron
                        pass
                    elif n2.is_input_neur or n2.bias:  # input can't be the second neuron
                        pass
                    elif n1 == n2:  # can't be the same neuron
                        pass
                    else:
                        # neurons can't be connected recursively
                        rec = False
                        for c in n1.out_connections:
                            if c in n2.in_connections:
                                rec = True
                                break
                        for c in n2.out_connections:
                            if c in n1.in_connections:
                                rec = True
                                break
                        if not rec:
                            offspring.connections.append(Connection(n1, n2))

                if random.random() < CHANCE_REMOVE_CONNECTION:
                    if len(offspring.connections) > 0:
                        conn = random.choice(offspring.connections)
                        conn.terminate()
                        offspring.connections.remove(conn)

                if random.random() < CHANCE_ADD_NEURON:
                    if len(offspring.connections) > 0:
                        conn = random.choice(offspring.connections)
                        n1, n2, w = conn.from_neuron, conn.to_neuron, conn.weight
                        conn.terminate()
                        offspring.connections.remove(conn)
                        n3 = Neuron()
                        offspring.h_neurons.append(n3)
                        offspring.connections.append(Connection(n1, n3, 1))
                        offspring.connections.append(Connection(n3, n2, w))

                if random.random() < CHANCE_REMOVE_NEURON:
                    not_connected = []
                    for neuron in offspring.h_neurons:
                        if len(neuron.in_connections) + len(neuron.out_connections) == 0:
                            not_connected.append(neuron)
                    if len(not_connected) != 0:
                        to_remove = random.choice(not_connected)
                        offspring.h_neurons.remove(to_remove)

        return offspring

    def get_image_of_network(self):
        " image"
        width = 500
        height = 350
        x_inp = 20
        surf = pygame.Surface((width, height))
        surf.fill(GREEN)

        dist = height // len(self.input_neurons)
        not_drawed = len(self.input_neurons)
        for y_inp in range(0, height, dist):
            not_drawed -= 1
            pygame.draw.rect(surf, GRAY, pygame.Rect(x_inp, y_inp, 10, 10))
        return surf

    def __str__(self):
        data = self.get_structure()
        return '\n' + str(list(map(str, data[0]))) + '\n' + \
               str(list(map(str, data[1]))) + '\n' + \
               str(list(map(str, data[2]))) + '\n' + \
               str(list(map(str, data[3]))) + '\n'


class Agent():
    def __init__(self, state_size, action_size, network=None):
        self.fitness = 0
        self.state_size = state_size
        self.action_size = action_size
        if network == None:
            self.network = Network(state_size, action_size)
        else:
            self.network = network

    def act(self, s):
        act_values = self.network.predict(s)
        return act_values

    def create_offspring(self):
        return Agent(self.state_size, self.action_size, self.network.create_offspring())


def solve_cart_pole():
    def evaluate_individual(individual: Agent, env):
        state = env.reset()
        r = 0
        for i in range(500):  # session length
            # env.render()
            actions = individual.act(state)
            state, reward, done, info = env.step(np.argmax(actions))
            r += reward
            if done:
                # print(r, 'reward')
                return r

    import gym
    population_amount = 100
    generations_to_run = 40
    percent_of_elitism = 0.4

    env = gym.make('MountainCar-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # print(state_size,action_size)

    population = [Agent(state_size, action_size) for i in range(population_amount)]

    def fitness_data(population):
        mx = -1000
        su = 0
        for agent in population:
            if agent.fitness > mx: mx = agent.fitness
            su += agent.fitness
        return round(su / len(population), 2), mx

    for i_generation in range(generations_to_run):
        for agent in population:  # evaluate
            agent.fitness = evaluate_individual(agent, env)
        for i in range(round(population_amount - (population_amount * percent_of_elitism))):
            a1, a2 = random.sample(population, 2)
            if a1.fitness > a2.fitness:
                population.remove(a2)
                population.append(a1.create_offspring())
            elif a2.fitness > a1.fitness:
                population.remove(a1)
                population.append(a2.create_offspring())
        print('Generation {}; Avg fitness {}; Max fitness {}'.format(str(i_generation),
                                                                     *list(map(str, fitness_data(population)))))
    mx, ag = -1000, None
    fle = open('log\structures_of_final_generation.txt', 'w')
    for agent in population:
        fle.write(str(agent.network))
        if agent.fitness > mx:
            mx = agent.fitness
            ag = agent
    fle.close()
    print(ag.network)


def test():
    n = Network(4, 2)
    # print(n)
    p = n.predict((0.2, 0.4, -1, 0.8))
    # print(p)
    n2 = n.create_offspring()
    # print(n2)
    p = n2.predict((0.2, 0.4, -1, 0.8))
    # print(p)

    pygame.init()
    display_surf = pygame.display.set_mode((800, 600))
    timer = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
        s = n2.get_image_of_network()
        display_surf.blit(s, (0, 0))

        timer.tick(60)


if __name__ == '__main__':
    solve_cart_pole()
    # test()
