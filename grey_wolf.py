import numpy as np
import random

from gwo_oscar import oscar_agent

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Grey Wolf Optimizer Functions
def initial_position(pack_size=5, min_values=[0.0001, 0.0001, 0.9, 0.005, 32], max_values=[0.01, 0.01, 0.99, 0.1, 128], target_function=None):
    position = np.zeros((pack_size, len(min_values) + 1))
    for i in range(pack_size):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, :-1])
    return position

def alpha_position(dimension=5, target_function=None):
    alpha = np.zeros((1, dimension + 1))
    for j in range(dimension):
        alpha[0, j] = 0.0
    alpha[0, -1] = target_function(alpha[0, :-1])
    return alpha

def beta_position(dimension=5, target_function=None):
    beta = np.zeros((1, dimension + 1))
    for j in range(dimension):
        beta[0, j] = 0.0
    beta[0, -1] = target_function(beta[0, :-1])
    return beta

def delta_position(dimension=5, target_function=None):
    delta = np.zeros((1, dimension + 1))
    for j in range(dimension):
        delta[0, j] = 0.0
    delta[0, -1] = target_function(delta[0, :-1])
    return delta

def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(position.shape[0]):
        if updated_position[i, -1] < alpha[0, -1]:
            alpha[0, :] = np.copy(updated_position[i, :])
        if alpha[0, -1] <= updated_position[i, -1] < beta[0, -1]:
            beta[0, :] = np.copy(updated_position[i, :])
        if beta[0, -1] <= updated_position[i, -1] < delta[0, -1]:
            delta[0, :] = np.copy(updated_position[i, :])
    return alpha, beta, delta

def update_position(position, alpha, beta, delta, a_linear_component=2, min_values=[0.0001, 0.0001, 0.9, 0.005, 32], max_values=[0.01, 0.01, 0.99, 0.1, 128], target_function=None):
    updated_position = np.copy(position)
    for i in range(position.shape[0]):
        for j in range(len(min_values)):
            r1_alpha = random.random()
            r2_alpha = random.random()
            a_alpha = 2 * a_linear_component * r1_alpha - a_linear_component
            c_alpha = 2 * r2_alpha
            distance_alpha = abs(c_alpha * alpha[0, j] - position[i, j])
            x1 = alpha[0, j] - a_alpha * distance_alpha

            r1_beta = random.random()
            r2_beta = random.random()
            a_beta = 2 * a_linear_component * r1_beta - a_linear_component
            c_beta = 2 * r2_beta
            distance_beta = abs(c_beta * beta[0, j] - position[i, j])
            x2 = beta[0, j] - a_beta * distance_beta

            r1_delta = random.random()
            r2_delta = random.random()
            a_delta = 2 * a_linear_component * r1_delta - a_linear_component
            c_delta = 2 * r2_delta
            distance_delta = abs(c_delta * delta[0, j] - position[i, j])
            x3 = delta[0, j] - a_delta * distance_delta

            position[i, j] = (x1 + x2 + x3) / 3  # Update position

            # Ensure the position stays within the allowed bounds
            position[i, j] = np.clip(position[i, j], min_values[j], max_values[j])

        # Update fitness value
        position[i, -1] = target_function(position[i, :-1])

    return updated_position

def target_function(variables_values):
    actor_lr = variables_values[0]
    critic_lr = variables_values[1]
    gamma = variables_values[2]
    tau = variables_values[3]
    batch_size = int(variables_values[4])

    # Initialize DDPG agent with the hyperparameters
    avg_reward = oscar_agent(state_dim=3, action_dim=1, actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau, batch_size=batch_size)
    
    return -avg_reward  # Negative reward for optimization

def gwo_optimizer():
    # Define bounds for hyperparameters
    min_values = [0.0001, 0.0001, 0.9, 0.005, 32]
    max_values = [0.01, 0.01, 0.99, 0.1, 256]

    # Number of wolves and iterations
    pack_size = 5
    iterations = 50

    # Initialize positions of wolves
    position = initial_position(pack_size=pack_size, min_values=min_values, max_values=max_values, target_function=target_function)

    # Initialize alpha, beta, and delta
    alpha = alpha_position(dimension=len(min_values), target_function=target_function)
    beta = beta_position(dimension=len(min_values), target_function=target_function)
    delta = delta_position(dimension=len(min_values), target_function=target_function)

    # Main loop for GWO
    for iteration in range(iterations):
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position = update_position(position, alpha, beta, delta, a_linear_component=2, min_values=min_values, max_values=max_values, target_function=target_function)

        print(f"Iteration {iteration + 1}/{iterations}, Best Reward: {-alpha[0, -1]}")

    print(f"Best hyperparameters: {alpha[0, :-1]}")
    return alpha[0, :-1]  # Return the best hyperparameters

gwo_optimizer()
