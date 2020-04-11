# -*- coding: utf-8 -*-


import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score




def initialize_random_chromosome(t):
    """
    Create a random individual
    """
    vector = [random.randint(0, 100) for i in range(t)]
    return list(vector)



def create_initial_population(N, t):
    """
        Create the initial population (IP)
    """

    return np.asarray([initialize_random_chromosome(t) for i in range(N)])
    


# =============================================================================
#   EVALUATION
# =============================================================================

def calculate_fitness(individual, C, l ,M):    
    fitness=0.0
    for index, row in C.iterrows():
        union = list(zip(row.tolist(), individual))
        #To create a key-value mmap with the sum of the weights
        label_sum = dict()
        for k, v in union:
            label_sum[k] = label_sum.get(k, 0) + v
            

        predicted = sorted(label_sum.items(), key = lambda x:x[1], reverse =True)[0][0]
        real = l[index]
        #dataFrame.loc[<ROWS RANGE> , <COLUMNS RANGE>]
        cost = M.iloc[real,predicted]
        
        fitness = fitness + cost
        
        
    return fitness



     


# =============================================================================
#  UNIFORM CROSSOVER
# =============================================================================
def uniform_crossover(parents, n_children, t):
    
    population = list()
    for i in range(n_children):
        children = []
        for i in range(t):
            eleccion = random.randrange(2) #The father is chosen
            children.append(parents[eleccion][i]) 
        population.append(children)
    return population





# =============================================================================
#  TOURNAMENT SELECTION 
# =============================================================================
def tournament_selection(population,  C, l ,M, sel_pressure, t, parents_number, n_children, N):
    """
        Puntua todos los elementos de la poblacion (population) y se queda con algunos individuos
        mediante la selección por torneo.  
    """   
    scored = [(calculate_fitness(i, C, l ,M), list(i)) for i in population] #It calculates the fitness of each individual in the population
    ordered = sorted(scored, key=lambda x:x[0])

    # The first two best are chosen to leave them in the next generation
    final_population = list()
    
    for i in range(parents_number):
        final_population.append(ordered[i][1])

    while len(final_population)<N:
        randomized = random.sample(scored, sel_pressure)

        randomized = sorted(randomized, key=lambda x:x[0])
        

        parents = [randomized[0][1], randomized[1][1]]
        H = uniform_crossover(parents, n_children, t)
        for elem in H:
            final_population.append(elem)
            
    ordered_population_by_fitness = sorted([(calculate_fitness(i, C, l ,M), list(i)) for i in final_population], key=lambda x:x[0]) 





    
    best_two = [elem[1] for elem in ordered_population_by_fitness[:parents_number]]

            
    return best_two, ordered_population_by_fitness[0][0], final_population  






## =============================================================================
##  UNIFORM CROSSOVER MUTATION
## =============================================================================
def mutate_population(population, parents_number, mu_pressure, t):
    """
        Individuals are mutated at random. Without the mutation of new genes, 
        the solution could never be reached.
    """    
    for i in range(len(population)-parents_number):
        if random.random() <= mu_pressure:
            #Each individual in the population (except parents) has a probability of mutating
            point = random.randint(0, t-1) # A random poin is chosen randomly
            new_value = random.randint(1,100) #New value for that poing
            
            #to ensure that the new value does not match the old one
            while new_value == population[i+parents_number][point]:
                new_value = random.randint(1,100)
                
            #Mutation is applied           
            population[i+parents_number][point] = new_value                    

    
    return population

  
## =============================================================================
##  MAIN
## =============================================================================
    
def genetic_algorithm(C,l, M, sel_pressure, t, parents_number, n_children, N, mu_pressure, G):
    #Initialize the population
    IP = create_initial_population(N, t)
    padres, fitness, FP = tournament_selection(IP, C,l, M, sel_pressure, t, parents_number, n_children, N)
    fitness_evolucion = list()
    fitness_evolucion.append(fitness)


    for i in range(G):
        padres, fitness, FP = tournament_selection(FP, C,l, M, sel_pressure, t, parents_number, n_children, N)
        fitness_evolucion.append(fitness)  
        FP = mutate_population(FP, parents_number, mu_pressure, t)
    
        
    return padres[0], fitness_evolucion[-1]
    



    
    


    

