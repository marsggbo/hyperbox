import numpy as np
import random


def dominates(a: np.array, b: np.array):
    """
    Returns True if individual a dominates individual b, False otherwise.
    The smaller, the better.
    """
    return np.all(a <= b) and not np.all(a == b)

def non_dominated_sort(fitnesses: np.ndarray):
    """
    Performs non-dominated sorting on a set of fitness matrix with shape of (num_samples, num_objectives).
    Returns a list of fronts, where each front is a list of indices into the fitness values array.
    """
    population_size = len(fitnesses) # number of samples
    fronts = [[]] # initialize the pareto-front list
    dominating_set = [set() for _ in range(population_size)] # dominating_set[i] means the set of individuals that i-th individual dominates
    num_dominated = np.zeros(population_size) # num_dominated[i] means the number of individuals that dominate i-th individual
    
    # Compute the domination relationships for each individual in the population
    for p in range(population_size):
        for q in range(population_size):
            if dominates(fitnesses[p], fitnesses[q]):
                dominating_set[p].add(q)
            elif dominates(fitnesses[q], fitnesses[p]):
                num_dominated[p] += 1
        # If individual p is not dominated by any other individual, add it to the current Pareto front
        if num_dominated[p] == 0:
            fronts[-1].append(p)

    # Build subsequent Pareto fronts
    while len(fronts[-1]) > 0:
        new_front = []
        for p in fronts[-1]:
            for q in dominating_set[p]:
                num_dominated[q] -= 1
                # If individual q is no longer dominated by any other individual,
                # add it to the new Pareto front
                if num_dominated[q] == 0:
                    new_front.append(q)
        fronts.append(new_front)

    return fronts[:-1] # remove the last empty front

def crowding_distance(fitnesses: np.ndarray, front: list):
    """
    Computes the crowding distance for each individual in a front.
    Returns an array of crowding distances.
    """
    distances = np.zeros(len(fitnesses))
    num_objectives = fitnesses.shape[1]
    
    # Calculate crowding distance for each objective
    for m in range(num_objectives):
        # Sort the front based on the objective m values
        sorted_front = sorted(front, key=lambda i: fitnesses[i][m])

        # Assign infinite crowding distance to boundary individuals to ensure they are always selected
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')
        
        # Calculate the minimum and maximum fitnesses for objective m
        f_min = fitnesses[sorted_front[0]][m]
        f_max = fitnesses[sorted_front[-1]][m]
        
        # Calculate crowding distance for intermediate individuals
        for i in range(1, len(sorted_front) - 1):
            distances[sorted_front[i]] += (fitnesses[sorted_front[i+1]][m] - fitnesses[sorted_front[i-1]][m]) / (f_max - f_min)
    
    return distances

def nsga2_select(fitnesses: np.array, num_selection: int):
    # Perform non-dominated sorting
    fronts = non_dominated_sort(fitnesses)
    
    selected_indices = []
    # Iterate through fronts
    for front in fronts:
        # If adding the entire front does not exceed the selection limit, 
        # add the whole front to the selected indices
        if len(selected_indices) + len(front) <= num_selection:
            selected_indices.extend(front)
        else:
            # Compute crowding distances
            distances = crowding_distance(fitnesses, front)
            
            # Sort indices by descending crowding distance
            front_sorted_by_crowding = sorted(front, key=lambda i: distances[i], reverse=True)
            
            # Select remaining individuals
            remaining = num_selection - len(selected_indices)
            selected_indices.extend(front_sorted_by_crowding[:remaining])
            break
    
    # Return the indicies of selected individuals
    return selected_indices


if __name__ == '__main__':
    fitnesses = np.array([
        #  1/acc, model size
        [1/90, 15],
        [1/80,65],
        [1/95,16],
        [1/98,10],
        [1/60,66],
        [1/80,96]
    ])
    population = {i: f"model{i}" for i in range(len(fitnesses))}
    selected = nsga2_select(population, fitnesses, 3)
    print(selected)