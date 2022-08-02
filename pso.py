# ------------------------------------------------------------------------------+
#   code from:
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#   Modify by:
#   Nicolás Caselli
#   Agoust, 2022
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import time

from scripts.utils import *
import numpy as np
from cec2013lsgo.cec2013 import Benchmark
import csv
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
    def __init__(self, ObjetiveFunction, NP, D, bound_min, bound_max, N_Gen, num_function, ejecution, BKS,  name_logs_file='logs.csv'):

        global num_dimensions
        num_dimensions = D
        self.err_best_g = -1
        self.num_function = num_function
        self.ejecution = ejecution
        self.BKS = BKS
        self.seed = int(time.time())

        # MH params
        self.function = ObjetiveFunction
        self.NP = NP #number particles
        self.D = D
        self.N_Gen = N_Gen

        self.Lower = bound_min
        self.Upper = bound_max
        self.F_min = 0
        self.improve_percentage = 1.0

        self.X = []
        self.fitness = []



    def init_particleSwarm(self):
        '''
        Initialize the variables of Cuckoo Search
        '''
        np.random.seed(self.seed)

        for i in range(self.NP):
            x = (self.Upper - self.Lower) * np.random.rand(self.D,) + self.Lower
            self.X.append(x)
            self.fitness.append(self.function(x))

        self.X = np.array(self.X).copy()
        self.best = self.X[0, :].copy()
        self.F_min = self.function(self.best)
    def execute(self, name_logs_file='logs.csv', name_cluster_logs_file='clusters.csv', original_MH=True, interval_logs=100):
        '''
        Ejecuta la MH con los parámetros creados en el constructor
        :param name_logs_file:
        :param name_cluster_logs_file:
        :param original_MH:
        :param interval_logs:
        :return:
        '''

        err_best_g = -1  # best error for group
        pos_best_g = []  # best position for group

        # Parametros de log
        logs_file = open(name_logs_file, mode='w')
        initial_time = time.perf_counter()
        logs_writter = csv.writer(logs_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        logs_writter.writerow(
            'function,ejecution,iteration,D,NP,N_Gen,A,r,fmin,fmax,lower,upper,alpha,gamma,time_ms,seed,BKS,fitness,%improvement'.split(
                ','))

        # establish the swarm
        self.init_particleSwarm()
        swarm = []
        for i in range(0, self.NP):
            swarm.append(Particle(self.X))

        # begin optimization loop
        i = 0
        while i < self.N_Gen:
            if (i % 100 == 0):
                # Logs purposes
                MH_params = f'{self.D},{self.NP},{self.N_Gen},{self.pa},{self.beta}'
                MH_params += f',{self.Lower},{self.Upper}'
                current_time = parseSeconds(time.perf_counter() - initial_time)
                log = f'{self.num_function},{self.ejecution},{t},{MH_params},{current_time},{self.seed},{self.BKS},"{self.F_min}","{self.improve_percentage}"'
                logs_writter.writerow(log.split(','))
                print('\n' + log)
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, self.NP):
                swarm[j].evaluate(self.function)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, self.NP):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(self.Lower, self.Upper)
            print(f'ITER:{i + 1} - err_best_g:{err_best_g}')
            # print(f'pos_best_g:{pos_best_g}')

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