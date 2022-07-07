# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import numpy as np
from cec2013lsgo.cec2013 import Benchmark
import math


# --- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total


# --- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self, x0):
        self.position_i = np.array([])  # particle position
        self.velocity_i = np.array([])  # particle velocity
        self.pos_best_i = np.array([])  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual

        for i in range(0, num_dimensions):
            self.velocity_i = np.append(self.velocity_i, random.uniform(-1, 1))
            self.position_i = np.append(self.position_i, x0[i])
    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognative constant
        c2 = 2  # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bound_min, bound_max):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bound_max:
                self.position_i[i] = bound_max

            # adjust minimum position if neseccary
            if self.position_i[i] < bound_min:
                self.position_i[i] = bound_min


class PSO():
    def __init__(self, costFunc, x0, bound_min, bound_max, num_particles, maxiter):
        '''
        Ejecuta algoritmo de optimización PSO acorde a los parámetros ingresados.
        :param costFunc: funcion de coste a minimizar
        :param x0: valores iniciales de enjambre
        :param bound_min: límite inferior de valores
        :param bound_max: límite maximo de valores
        :param num_particles: número de particulas en el enjambre
        :param maxiter: número máximo de iteraciones.
        '''
        global num_dimensions


        num_dimensions = len(x0)
        err_best_g = -1  # best error for group
        pos_best_g = []  # best position for group

        # establish the swarm
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i = 0
        while i < maxiter:
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bound_min, bound_max)
            print(f'ITER:{i+1} - err_best_g:{err_best_g}')
            #print(f'pos_best_g:{pos_best_g}')

            i += 1

        # print final results
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)



# --- RUN ----------------------------------------------------------------------+
if __name__ == '__main__':
    bench = Benchmark()
    info = bench.get_info(1)
    min_bound = info['lower']
    max_bound = info['upper']
    initial = np.random.uniform(low=min_bound, high=max_bound, size=info['dimension'])  # initial starting location [x1,x2...]
    ObjetiveFunction = bench.get_function(1)
    print(f"inicial = {len(initial)}"
          f"\ninfo = {info}"
          f"\nObjetiveFunction = {ObjetiveFunction}"
          f"\nmin_bound = {min_bound} "
          f"\nmax_bound = {max_bound}")
    print(f"Evaluando en función {1}= {ObjetiveFunction(initial)}")
    #bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    PSO(ObjetiveFunction, initial, min_bound, max_bound, num_particles=15, maxiter=2500)

# --- END ----------------------------------------------------------------------+