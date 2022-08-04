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
from scripts.utils import *
import random
import time


import numpy as np
from cec2013lsgo.cec2013 import Benchmark
import csv
import math

MAX_PARTICLES = 100
MIN_PARTICLES = 10
INCREMENTS_PARTICLES = 2
INCREMENTS_PARTICLES_PER_CLUSTER = 2         # Cantidad de nidos a agregar por cluster
IMPROVE_PERCENTAGE_ACCEPTED = 10        # Porcentaje de mejora aceptado para aplicar el autonomo
DIFF_CLUSTER_PERCENTAGE_ACCEPTED = 5    # Diferencia porcentual aceptado para clusters juntos

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
        self.N_Gen = N_Gen #number of generations, or number of iterations

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

    def calculate_percentage(self, past_best, new_best):
        '''
        Calcula el porcentaje de mejora a partir del mejor actual y el pasado mejor
        '''
        current_difference = past_best - new_best
        percentage = (current_difference * 100) / past_best
        return percentage

    def check_improve(self, clusters_writter, past_best, iteration):
        '''
        Aplica el autonomo
        '''
        global INCREMENTS_PARTICLES
        global INCREMENTS_PARTICLES_PER_CLUSTER

        # Se revisa si el porcentaje de mejora es menor que el aceptado, si lo es
        # se implementan las estrategias de autoajuste
        if self.improve_percentage > IMPROVE_PERCENTAGE_ACCEPTED:
            # Si la solucion ha mejorado, y no se ha llegado al limite se decrementan los murcielagos
            if self.NP - INCREMENTS_PARTICLES >= MIN_PARTICLES:
                # Se decrementan la cantidad de murcielagos
                self.NP -= INCREMENTS_PARTICLES

                # Se eliminan los peores murcielagos con sus datos de cada lista
                self.fitness = self.fitness[:-INCREMENTS_PARTICLES]
                self.X = self.X[:-INCREMENTS_PARTICLES]
        else:
            print(f"Improvement percetage: {round(self.improve_percentage, 2)}%  Applying self-tunning strategies")

            new_solutions = []

            # Se clusterizan las soluciones
            k = 3
            clusters, epsilon = clusterize_solutions(self.X, k)
            labels = clusters.labels_
            unique_labels = np.unique(labels)
            cant_clusters = unique_labels.shape[0]

            # Se obtiene la informacion de los clusters
            info_clusters = getInfoClusters(labels, self.fitness)

            # VER getInfoClusters PARA VER EL FORMATO DE info_clusters
            # Se guardan los logs del cluster
            for label in unique_labels:
                min_value = info_clusters[label]['min']
                max_value = info_clusters[label]['max']
                mean_cluster = info_clusters[label]['mean']
                quantity = info_clusters[label]['quantity']

                cluster_logs = f'{self.seed},{self.num_function},{self.ejecution},{iteration},{cant_clusters},{min_value},{max_value},{quantity},{mean_cluster},{epsilon},{k},{label}'
                clusters_writter.writerow(cluster_logs.split(','))

            # Sino se alcanzo el limite, se incrementa la poblacion de murcielagos
            if self.NP + (cant_clusters * INCREMENTS_PARTICLES_PER_CLUSTER) < MAX_PARTICLES:
                # Se obtienen las nuevas soluciones generadas (llega una lista de tuplas, que guarda
                # como primer elemento la solucion generada localmente, y como segundo elemento el indice
                # del murcielago sobre el que se genero la solucion local
                new_solutions = self.increment_cluster(clusters)

                # Se guarda la cantidad de murcielagos que se agregaron, para despues eliminar la misma cantidad
                INCREMENTS_PARTICLES = cant_clusters * INCREMENTS_PARTICLES_PER_CLUSTER

            # Si todos los muercielagos estan muy juntos, se reemplaza la mitad
            self.replace_cluster(clusters)

            # Si hay nuevas soluciones se agregan
            for element in new_solutions:
                nest, index = element
                self.add_new_nest(nest, index)

            # Se actualiza el mejor fitness
            self.best_nest()
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
        logs_writter.writerow('function'
                              ',ejecution'
                              ',iteration'
                              ',D'
                              ',NP'
                              ',N_Gen'
                              ',lower'
                              ',upper'
                              ',time_ms'
                              ',seed'
                              ',BKS'
                              ',fitness'
                              ',%improvement'
                              ''.split(','))
        # Archivo de logs de los clusters
        clusters_file = open(name_cluster_logs_file, mode='w')
        cluster_writter = csv.writer(clusters_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cluster_writter.writerow(
            'seed,function,ejecution,iteration,cantClusters,min_value,max_value,cantElements,meanCluster,epsilon,k,label'.split(
                ','))

        # establish the swarm
        self.init_particleSwarm()
        swarm = []
        for i in range(0, self.NP):
            swarm.append(Particle(self.X[i]))

        # begin optimization loop
        i = 0
        while i <= self.N_Gen:
            if (i % 100 == 0):
                # Logs purposes

                MH_params = f'{self.D},{self.NP},{self.N_Gen}'
                MH_params += f',{self.Lower},{self.Upper}'
                current_time = parseSeconds(time.perf_counter() - initial_time)
                log = f'{self.num_function},{self.ejecution},{i},{MH_params},{current_time},{self.seed},{self.BKS},"{self.F_min}","{self.improve_percentage}"'
                logs_writter.writerow(log.split(','))
                print('\n' + log)

                if i != 0:
                    if not original_MH:
                        # Se ajusta la cantidad de soluciones dependiendo del desempeño
                        self.check_improve(cluster_writter, past_best, i)

                    past_best = self.F_min
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, self.NP):
                swarm[j].evaluate(self.function)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    self.improve_percentage = self.calculate_percentage(err_best_g, swarm[j].err_i)
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)
                    self.F_min = err_best_g


            # cycle through swarm and update velocities and position
            for j in range(0, self.NP):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(self.Lower, self.Upper)
            #print(f'ITER:{i + 1} - err_best_g:{err_best_g}')
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
    print(initial)
    print(f"inicial = {len(initial)}"
          f"\ninfo = {info}"
          f"\nObjetiveFunction = {ObjetiveFunction}"
          f"\nmin_bound = {min_bound} "
          f"\nmax_bound = {max_bound}")
    print(f"Evaluando en función {1}= {ObjetiveFunction(initial)}")
    print('X:\n')
    SV = []
    x = (max_bound - min_bound) * np.random.rand(5,) + min_bound
    SV.append(x)
    x = (max_bound - min_bound) * np.random.rand(5, ) + min_bound
    SV.append(x)
    x = (max_bound - min_bound) * np.random.rand(5, ) + min_bound
    SV.append(x)
    print(SV)
    #bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    #PSO(ObjetiveFunction, initial, min_bound, max_bound, num_particles=15, maxiter=2500)

# --- END ----------------------------------------------------------------------+