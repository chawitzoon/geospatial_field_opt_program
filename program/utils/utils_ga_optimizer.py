import numpy as np
import json

import random
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from scipy.stats import norm
import sys

class GA(object):
    def __init__(self, optimized_var_range, optimized_var_len, output_func, pop_size = 8, num_generations = 5, parent_ratio = 0.5, extremum = 'maximize', save_fname='test'):

        '''
        ga optimizer class
        param: optimized_var_range: list of list, list as [[min_value_1, max_value_1], [min_value_2, max_value_2],..]
        param: optimized_var_len: int, length of value to be optimized
        param: output_func: python function returning the value to be optimized
        param: pop_size: int, population size per generation (default = 8)
        param: num_generations: int, number of iterations (default = 5)
        param: parent_ratio: float, ratio between parent:pop in range (0,1)
        param: extremum: string, either 'maximize' or 'minimize'
        param: save_fname: string, name of the run for plotting and also used in save_result function (default='test')
        '''
        self.optimized_var_num = optimized_var_len
        self.optimized_var_range = optimized_var_range
        self.output_func = output_func
        self.pop_size = pop_size
        self.num_generation = num_generations
        self.extremum = extremum

        # population pool
        self.new_population = np.empty(shape=(self.pop_size, self.optimized_var_num))

        for j in range(self.optimized_var_num):
            self.new_population[:,j] = np.around(np.random.uniform(low=optimized_var_range[j][0], high=optimized_var_range[j][1], size=(self.pop_size,)), decimals=0)

        self.parents_num = np.uint8(pop_size*parent_ratio)
        self.buffer_fitness_dict = {}
        # print('finish creating populations')

        # output
        self.fig = None
        self.save_fname = save_fname
        self.best_fitness = None
        self.best_gene = None

        
    def cal_pop_fitness(self, output_func, pop):
        fitness_list = []
        for solution in pop:
            solution = solution.astype(int)
            solution_str=str(solution)
            if solution_str in self.buffer_fitness_dict.keys():
                fitness_list.append(self.buffer_fitness_dict[solution_str])
            else:
                fitness = output_func(np.array(solution))

                fitness_list.append(fitness)
                self.buffer_fitness_dict[solution_str] = fitness
                
        self.fitness = np.array(fitness_list)
        return self.fitness

    def select_mating_pool(self, fitness): # can set size by self.parents_num

        parents = np.empty(shape=(self.parents_num, self.optimized_var_num))

        if self.extremum == 'maximize':
            sort_idx_list = (-np.array(fitness)).argsort()[:self.parents_num] 
        elif self.extremum == 'minimize':
            sort_idx_list = (np.array(fitness)).argsort()[:self.parents_num] 
  
        for count, i in enumerate(sort_idx_list):
            parents[count,:] = self.new_population[i,:]

        return parents


    def crossover(self, parents):
        offspring = np.empty(shape=(self.pop_size - len(parents), self.optimized_var_num))
        crossover_point = np.random.randint(self.optimized_var_num) # random the crossover point

        for k in range(self.pop_size - len(parents)):

            # Index of the first parent to mate.
            parent1_idx = k%len(parents)

            # Index of the second parent to mate.
            parent2_idx = (k+1)%len(parents)

            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

            # The new offspring will have its second half of its genes taken from the second parent.
            # offspring.append(new_offspring)
        
        return offspring

    def mutation(self, offspring):
        # Mutation changes a single gene in each offspring randomly.
        for new_offspring in offspring:
            # random a var to be mutated
            idx = np.random.randint(self.optimized_var_num)
            new_offspring[idx] = np.around(np.random.uniform(low=self.optimized_var_range[idx][0], high=self.optimized_var_range[idx][1]), decimals=0)
        return offspring


    def update_pop(self, plot=False):
        fig = plt.figure()
        plt.axis([0, self.num_generation, None, None])
        x_array = [i for i in range(self.num_generation)]
        mean_fitness = np.zeros((self.num_generation))
        std_fitness = np.zeros((self.num_generation))
        for iter in range(self.num_generation):
            # if iter % 1 == 0:
            #     print(f'iteration {iter}/{self.num_generation}')
            fitness = self.cal_pop_fitness(self.output_func, self.new_population)
            for fitness_one in fitness:
                plt.scatter(iter, fitness_one)
            mu, std = norm.fit(fitness)
            mean_fitness[iter] = mu
            std_fitness = std

            parents = self.select_mating_pool(fitness)

            offspring = self.crossover(parents)
            offspring = self.mutation(offspring)


            # Creating the new population based on the parents and offspring.
            # self.new_population = parents + offspring
            self.new_population[:self.parents_num, :] = parents
            self.new_population[self.parents_num:, :] = offspring

        if self.extremum == 'maximize':
            best_fitness_idx = np.argmax(np.array(fitness))

        elif self.extremum == 'minimize':
            best_fitness_idx = np.argmin(np.array(fitness))

        self.best_fitness = self.fitness[best_fitness_idx]
        self.best_gene = self.new_population[best_fitness_idx]
        
        plt.plot(x_array, mean_fitness)
        plt.fill_between(x_array, mean_fitness - std_fitness, mean_fitness + std_fitness, color='gray', alpha=0.2)
        plt.title(f"cost_function in the iteration of GA \n best value: {self.best_fitness} \n plan file: {self.save_fname}.json")

        self.fig = fig
        if plot:
            plt.show()

        return self.best_gene, self.best_fitness


    def save_result(self):
        if self.fig != None:
            self.fig.savefig(f'result_ga/{self.save_fname}.png', bbox_inches='tight')
            plt.close()

            with open(f'result_ga/{self.save_fname}.json', 'w') as fp:
                json.dump({'parameter': {'optimized_var_num': self.optimized_var_num, 
                                        'optimized_var_range': self.optimized_var_range,
                                        'pop_size': self.pop_size,
                                        'num_generation': self.num_generation,
                                        },
                            'best_gene': self.best_gene.tolist(),
                            'best_fitness': self.best_fitness,
                            'best_gene_list': self.new_population.tolist(),
                            'best_fitness_list': list(self.fitness)}, fp)
