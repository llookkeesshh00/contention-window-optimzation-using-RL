############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Grey Wolf Optimizer

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Grey_Wolf_Optimizer, File: Python-MH-Grey Wolf Optimizer.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Grey_Wolf_Optimizer>

############################################################################

# Required Libraries
import numpy  as np
import math
import random
import os


# Function
def target_function():
    return

# Function: Initialize Variables
def initial_position(pack_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((pack_size, len(min_values)+1))
    for i in range(0, pack_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

# Function: Initialize Alpha
def alpha_position(dimension = 2, target_function = target_function):
    alpha = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        alpha[0,j] = 0.0
    alpha[0,-1] = target_function(alpha[0,0:alpha.shape[1]-1])
    return alpha

# Function: Initialize Beta
def beta_position(dimension = 2, target_function = target_function):
    beta = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        beta[0,j] = 0.0
    beta[0,-1] = target_function(beta[0,0:beta.shape[1]-1])
    return beta

# Function: Initialize Delta
def delta_position(dimension = 2, target_function = target_function):
    delta =  np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        delta[0,j] = 0.0
    delta[0,-1] = target_function(delta[0,0:delta.shape[1]-1])
    return delta

# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(0, position.shape[0]):
        if (updated_position[i,-1] < alpha[0,-1]):
            alpha[0,:] = np.copy(updated_position[i,:])
        if (updated_position[i,-1] > alpha[0,-1] and updated_position[i,-1] < beta[0,-1]):
            beta[0,:] = np.copy(updated_position[i,:])
        if (updated_position[i,-1] > alpha[0,-1] and updated_position[i,-1] > beta[0,-1]  and updated_position[i,-1] < delta[0,-1]):
            delta[0,:] = np.copy(updated_position[i,:])
    return alpha, beta, delta

# Function: Updtade Position
def update_position(position, alpha, beta, delta, a_linear_component = 2, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    updated_position = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range (0, len(min_values)):   
            r1_alpha              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_alpha              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_alpha               = 2*a_linear_component*r1_alpha - a_linear_component
            c_alpha               = 2*r2_alpha      
            distance_alpha        = abs(c_alpha*alpha[0,j] - position[i,j]) 
            x1                    = alpha[0,j] - a_alpha*distance_alpha   
            r1_beta               = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_beta               = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_beta                = 2*a_linear_component*r1_beta - a_linear_component
            c_beta                = 2*r2_beta            
            distance_beta         = abs(c_beta*beta[0,j] - position[i,j]) 
            x2                    = beta[0,j] - a_beta*distance_beta                          
            r1_delta              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_delta              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_delta               = 2*a_linear_component*r1_delta - a_linear_component
            c_delta               = 2*r2_delta            
            distance_delta        = abs(c_delta*delta[0,j] - position[i,j]) 
            x3                    = delta[0,j] - a_delta*distance_delta                                 
            updated_position[i,j] = np.clip(((x1 + x2 + x3)/3),min_values[j],max_values[j])     
        updated_position[i,-1] = target_function(updated_position[i,0:updated_position.shape[1]-1])
    return updated_position

# GWO Function
def grey_wolf_optimizer(pack_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function):    
    count    = 0
    alpha    = alpha_position(dimension = len(min_values), target_function = target_function)
    beta     = beta_position(dimension  = len(min_values), target_function = target_function)
    delta    = delta_position(dimension = len(min_values), target_function = target_function)
    position = initial_position(pack_size = pack_size, min_values = min_values, max_values = max_values, target_function = target_function)
    while (count <= iterations):      
        print("Iteration = ", count, " f(x) = ", alpha[-1])      
        a_linear_component = 2 - count*(2/iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position           = update_position(position, alpha, beta, delta, a_linear_component = a_linear_component, min_values = min_values, max_values = max_values, target_function = target_function)       
        count              = count + 1       
    print(alpha[-1])    
    return alpha

######################## Part 1 - Usage ####################################

# Function to be Minimized (Six Hump Camel Back). Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def six_hump_camel_back(variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

gwo = grey_wolf_optimizer(pack_size = 15, min_values = [-5,-5], max_values = [5,5], iterations = 100, target_function = six_hump_camel_back)

# Function to be Minimized (Rosenbrocks Valley). Solution ->  f(x) = 0; xi = 1
def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value

gwo = grey_wolf_optimizer(pack_size = 100, min_values = [-5,-5], max_values = [5,5], iterations = 100, target_function = rosenbrocks_valley)
