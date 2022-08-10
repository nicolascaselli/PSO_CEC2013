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

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

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
    def __init__(self, x0, position_i=None, velocity_i=None, err_best_i=None, pos_best_i=None, costFunction=None):
        self.position_i = np.array([])  # particle position
        self.velocity_i = np.array([])  # particle velocity
        self.pos_best_i = np.array([])  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual
        if position_i is None:
            for i in range(0, num_dimensions):
                self.position_i = np.append(self.position_i, x0[i])
        else:
            self.position_i = position_i

        if velocity_i is None:
            for i in range(0, num_dimensions):
                self.velocity_i = np.append(self.velocity_i, random.uniform(-1, 1))
        else:
            self.velocity_i = velocity_i

        if pos_best_i is None:
            self.pos_best_i = np.array([])
        else:
            self.pos_best_i = pos_best_i

        if err_best_i is None:
            self.err_best_i = -1
        else:
            self.err_best_i = err_best_i
        if costFunction is not None:
            self.evaluate(costFunction)
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
        self.F_min = -1
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
        self.swarm = []

    def sort_by_fitness(self):
        '''
        sort all the particles by fitness (error_i)
        @:param l_swarm: list of particles
        @:return l_swarm: list of particles sorted by err_i
        '''

        self.swarm = sorted(self.swarm, key=lambda particle: particle.err_i)


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

    def update_improve_percentage(self, past_best):
        '''
        Actualiza el porcentaje de mejora
        '''
        self.improve_percentage = self.calculate_percentage(past_best, self.F_min)

    def check_improve(self, clusters_writter, past_best, iteration):
        '''
        Aplica el autonomo
        '''
        global INCREMENTS_PARTICLES
        global INCREMENTS_PARTICLES_PER_CLUSTER

        # Se revisa si el porcentaje de mejora es menor que el aceptado, si lo es
        # se implementan las estrategias de auto ajuste
        if self.improve_percentage > IMPROVE_PERCENTAGE_ACCEPTED:
            # Si la solucion ha mejorado, y no se ha llegado al límite se decrementan los murcielagos
            if self.NP - INCREMENTS_PARTICLES >= MIN_PARTICLES:
                # Se decrementan la cantidad de murcielagos
                self.NP -= INCREMENTS_PARTICLES

                # Se eliminan las peores particulas
                self.fitness = self.fitness[:-INCREMENTS_PARTICLES]
                self.swarm = self.swarm[:-INCREMENTS_PARTICLES]
                self.X = self.X[:-INCREMENTS_PARTICLES]
        else:
            print(f"Improvement percetage: {round(self.improve_percentage, 2)}%  Applying self-tunning strategies")

            new_solutions = []

            # Se clusterizan las soluciones
            k = 3

            for i in range(0, self.NP):
                self.X[i] = self.swarm[i].position_i
                self.fitness[i] = self.swarm[i].err_i

            clusters, epsilon = clusterize_solutions(self.X, k)
            labels = clusters.labels_
            unique_labels = np.unique(labels)
            cant_clusters = unique_labels.shape[0]

            # Se obtiene la información de los clusters
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

            # Si no se alcanzó el límite, se incrementa la población de partículas
            if self.NP + (cant_clusters * INCREMENTS_PARTICLES_PER_CLUSTER) < MAX_PARTICLES:
                # Se obtienen las nuevas soluciones generadas (llega una lista de tuplas, que guarda
                # como primer elemento la solución generada localmente, y como segundo elemento el índice
                # de la partícula sobre el que se generó la solución local
                new_solutions = self.increment_cluster(clusters)

                # Se guarda la cantidad de partículas que se agregaron, para después eliminar la misma cantidad
                INCREMENTS_PARTICLES = cant_clusters * INCREMENTS_PARTICLES_PER_CLUSTER

            # Si todas las partículas están muy juntas, se reemplaza la mitad
            self.replace_cluster(clusters)

            # Si hay nuevas soluciones se agregan
            for element in new_solutions:
                particle, index = element
                self.add_new_particle(particle, index)

            # Se actualiza el mejor fitness
            self.best_particle()

    def best_particle(self):
        '''
        Busca y actualiza la mejor partícula
        '''
        j = 0
        for i in range(self.NP):
            if self.fitness[i] < self.fitness[j]:
                j = i

        self.set_best_particle(self.X[j], self.fitness[j])

    def set_best_particle(self, particle, fitness):
        '''
        Establece la mejor partícula
        '''
        for i in range(self.D):
            self.best[i] = particle[i]

        self.F_min = fitness
    def replace_cluster(self, clusters):
        '''
        Reemplaza la mitad mas mala de los clusters, con soluciones generados aleatorias
        '''
        # Diccionario que contiene la informacion para calcular el promedio de cada cluster.
        # Para cada cluster se puede acceder a su informacion por su label,
        # como valor guarda un diccionario para organizar su informacion
        fitness_clusters = {l: {'sum': 0, 'total': 0} for l in np.unique(clusters.labels_)}

        # Se obtiene la suma de los fitness y el total de elementos en cada cluster
        for (index, label) in enumerate(clusters.labels_):
            fitness_clusters[label]['sum'] += self.fitness[index]
            fitness_clusters[label]['total'] += 1

        for label in fitness_clusters:
            # Se calcula el promedio
            suma = fitness_clusters[label]['sum']
            total = fitness_clusters[label]['total']
            mean_cluster = suma / total

            percentage_diff = self.calculate_percentage(self.F_min, mean_cluster)

            # if -1 <= self.F_min - mean_cluster <= 1:
            if -DIFF_CLUSTER_PERCENTAGE_ACCEPTED <= percentage_diff <= DIFF_CLUSTER_PERCENTAGE_ACCEPTED:
                # Se reemplaza la mitad mas mala del cluster con soluciones aleatorias usando la funcion de exploracion
                cant = total // 2

                for index in range(self.NP - 1, -1, -1):
                    if cant <= 0:
                        break

                    # Si el elemento actual pertenece al cluster que queremos repoblar
                    if clusters.labels_[index] == label:
                        self.X[index], self.fitness[index] = self.generate_random_solution(self.X[index])
                        self.swarm[index].err_i = self.fitness[index]
                        self.swarm[index].position_i = self.X[index]
                        cant -= 1

            print(percentage_diff, self.F_min - mean_cluster, self.F_min, mean_cluster, label)

    def generate_random_solution(self, solution):
        '''
        Genera una nueva solucion aleatoria para explorar el expacio de busqueda
        '''
        for j in range(self.D):
            random = np.random.uniform(0, 1)
            solution[j] = self.Lower + (self.Upper - self.Lower) * random

        fitness = self.function(solution)

        return solution, fitness
    def increment_cluster(self, clusters, Amean = None):
        '''
        Se incrementa la poblacion de los clusters, agregando particulas generados
        alrededor de los mejores de cada cluster
        '''
        if Amean == None:
            Amean = 1
        x_is_modified = False
        best_bat_clusters = {l: {'index': [], 'cant': 0} for l in np.unique(clusters.labels_)}

        # Se guardan los índices de los INCREMENTS_BATS_PER_CLUSTER mejores partículas de cada cluster
        for index, label in enumerate(clusters.labels_):
            if best_bat_clusters[label]['cant'] < INCREMENTS_PARTICLES_PER_CLUSTER:
                best_bat_clusters[label]['index'].append(index)
                best_bat_clusters[label]['cant'] += 1

        # Se guardan las nuevas soluciones generadas, junto con el índice de la partícula
        # sobre el cual se generó la solución local
        new_solutions = []

        # Se generan INCREMENTS_BATS_PER_CLUSTER soluciones locales de los mejores partículas de cada cluster
        for label in best_bat_clusters:
            for index in best_bat_clusters[label]['index']:
                # Se encuentra una nueva solucion local
                new_solution = np.empty(self.D)
                new_solution = self.generate_local_solution(new_solution, self.X[index], Amean)
                new_solutions.append((new_solution, index))

        return new_solutions

    def simple_bounds(self, value, lower, upper):
        '''
        Le aplica las bandas a 'value'
        '''
        if (value > upper):
            value = upper

        if (value < lower):
            value = lower

        return value
    def generate_local_solution(self, solution, particle, Amean):
        '''
        Genera una nueva solución local alrededor de la solución "particle"
        '''
        for j in range(self.D):
            random = np.random.uniform(-1.0, 1.0)
            solution[j] = self.simple_bounds(particle[j] + random * Amean, self.Lower, self.Upper)

        return solution
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
            'seed'
            ',function'
            ',ejecution'
            ',iteration'
            ',cantClusters'
            ',min_value'
            ',max_value'
            ',cantElements'
            ',meanCluster'
            ',epsilon'
            ',k'
            ',label'.split(','))

        # establish the swarm
        self.init_particleSwarm()

        for i in range(0, self.NP):
            self.swarm.append(Particle(self.X[i]))

        past_best = self.F_min
        # begin optimization loop
        i = 0
        while i <= self.N_Gen:
            #checks if the autonomous needs to intervene
            if (i % 100 == 0):
                #order the list of particle by fitness
                self.sort_by_fitness()
                self.update_improve_percentage(past_best)

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
            #end of autonomous section
            # print i,F_min
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, self.NP):
                self.swarm[j].evaluate(self.function)

                # determine if current particle is the best (globally)
                if self.swarm[j].err_i < err_best_g or err_best_g == -1:
                    self.improve_percentage = self.calculate_percentage(err_best_g, self.swarm[j].err_i)
                    pos_best_g = list(self.swarm[j].position_i)
                    err_best_g = float(self.swarm[j].err_i)
                    self.F_min = err_best_g

            # cycle through swarm and update velocities and position
            for j in range(0, self.NP):
                self.swarm[j].update_velocity(pos_best_g)
                self.swarm[j].update_position(self.Lower, self.Upper)
            #print(f'ITER:{i + 1} - F_min:{F_min}')
            # print(f'pos_best_g:{pos_best_g}')

            i += 1

        # print final results
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)

    def add_new_particle(self, new_particle, index):
        '''
        Agrega un nuevo murcielago a la poblacion
        '''
        self.X = np.append(self.X, [new_particle], axis=0)
        self.fitness.append(self.function(new_particle))
        self.NP += 1
        self.swarm.append(Particle(new_particle, velocity_i=self.swarm[index].velocity_i, costFunction=self.function))




# --- RUN ----------------------------------------------------------------------+
if __name__ == '__main__':
    '''
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
    #bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    #PSO(ObjetiveFunction, initial, min_bound, max_bound, num_particles=15, maxiter=2500)
    '''
    bench = Benchmark()
    info = bench.get_info(1)

    BKS = info['best']
    Lower = info['lower']
    Upper = info['upper']
    D = info['dimension']
    NP = 15
    N_Gen = 5000
    A = 0.95
    r = 0.1
    alpha = 0.9
    gamma = 0.5
    fmin = 0
    fmax = 1

    objetiveFunction = bench.get_function(1)
    name_ejecution_file = f'function{1}_{1}.csv'
    name_logs_file = 'Logs/' + name_ejecution_file
    name_cluster_file = 'Logs/clusters/' + name_ejecution_file

    swarm = PSO(objetiveFunction, NP, D, Lower, Upper, N_Gen, 1, 1, BKS)
    swarm.init_particleSwarm()
    # establish the swarm
    l_swarm = []
    for i in range(0, NP):
        l_swarm.append(Particle(swarm.X[i]))
    for i in range(0, NP):
        l_swarm[i].evaluate(objetiveFunction)

    #ordenamos el objeto por fitnes (err_i
    l_swarm = sorted(l_swarm, key=lambda particle: particle.err_i)
    solutions = []
    for i in range(0, NP):
        solutions.append(l_swarm[i].position_i)
    print(solutions)
    clusters, epsilon = clusterize_solutions(solutions, 3)
    labels = clusters.labels_
    unique_labels = np.unique(labels)
    cant_clusters = unique_labels.shape[0]
    print(f'\nClusters:{clusters.labels_}')
    print(f'\nepsilon:{epsilon}')
    print(f'\nlabels: {labels}')
    print(f'\nunique_labels: {unique_labels}')
    print(f'\ncant_clusters: {cant_clusters}')

# --- END ----------------------------------------------------------------------+