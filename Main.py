

import sys
from Instance import Instance
from objects import Test
import random
import os
from ConstructiveHeuristic import ConstructiveHeuristic
import time
from LocalSearches import tabuSearch , tabuSearch_capacity_simulation , tabuSearch_capacity
import numpy as np
from sim_learnheuristic import *
import pandas as pd
from objects import *






'''
The test is composed of the following parameters:
#Instance   Seed    Time    BetaBR    BetaLs  MaxIterLS
-Instance: Name the instance
-Seed: Seed used to generate random numbers in the BR heuristic
-Time: Maximum execution time
-BetaBR: Beta parameter used in the BR heuristic
-BetaLs: Beta parameter used in the Local search
'''
def readTest(testName):
    # Construct the file path using the provided test name
    fileName = 'test' + os.sep + testName + '.txt'
    
    # List to store the Test objects
    tests = []
    
    # Open the file for reading
    with open(fileName, 'r') as testsfile:
        # Iterate over each line in the file
        for linetest in testsfile:
            # Remove leading and trailing whitespaces from the line
            linetest = linetest.strip()
            
            # Check if the line is not a comment (does not contain '#')
            if '#' not in linetest:
                # Split the line using tab ('\t') as the delimiter
                line = linetest.split('\t')
                
                # Create a Test object using the values from the line
                test = Test(*line)
                
                # Append the Test object to the tests list
                tests.append(test)
    
    # Return the list of Test objects
    return tests



def stochastic_multistart(bestSol, t: Instance, cl):
    # Create an instance of the sim_learnheuristic with short simulation time and specified variance
    var = t.var
    small_simulation = sim_learnheuristic(t.short_simulation, var)

    # Perform simulations based on the chosen options
    small_simulation.simulation_2(bestSol, cl)

    elapsed = 0.0  # Initialize elapsed time
    iter = 0  # Initialize iteration counter
    elite_simulations = [bestSol]  # List to store elite solutions from the short simulation
    start = time.process_time()  # Record the start time

    while elapsed < t.Maxtime:
        iter += 1  # Increment iteration counter
        newSol, cl = heur.constructBRSol_capacity()  # Biased-randomized version of the heuristic
        newSol = tabuSearch_capacity(newSol, cl, t.maxIter, heur)  # Local Search (Tabu Search)
        whitebox.fit_logistic()
        #params_dict = {"max_time": t.Maxtime, "time": elapsed}
        #whitebox.fit_with_probability(whitebox.fit_logistic, t.prob, **params_dict)
        # Check if the new solution improves the best solution
        if newSol.of > bestSol.of:
            small_simulation.simulation_2(newSol, cl)

            if newSol.mean_stochastic_of["2"] >= bestSol.mean_stochastic_of["2"]:
                bestSol = newSol.copySol()  # Update Solution
                bestSol.time = elapsed
                elite_simulations.append(bestSol)

        elapsed = time.process_time() - start  # Update elapsed time

    large_simulation = sim_learnheuristic(t.long_simulation, var)  # Create an instance of the sim_learnheuristic with long simulation time

    # Perform simulations for the elite solutions from the short simulation
    for i in elite_simulations:
        large_simulation.simulation_2(i, cl)

    elite_simulations.sort(key=lambda x: x.stochastic_of["2"], reverse=True)
    return elite_simulations[0]  # Return the best solution from the elite solutions



def stochastic_multistart_simulation(bestSol, t: Instance):
    
    # Initialize the sim_learnheuristic object with short simulation time and specified variance
    var = t.var
    small_simulation = sim_learnheuristic(t.short_simulation, var)
    
    # Initial solution construction using biased-randomized heuristic
    bestSol, cl = heur.constructBRSol_capacity_simulation(sim_learnheuristic(20, var), 0.9)
    bestSol = tabuSearch_capacity_simulation(bestSol, cl, t.maxIter, heur, sim_learnheuristic(20, var), 0.9)
    
    # Perform the first simulation based on the chosen option (true for simulation_1)
    small_simulation.simulation_1(bestSol)
    
    
    
    # Initialize variables for time, iteration, and elite solutions
    elapsed = 0.0
    iter = 0
    elite_simulations = []
    elite_enter_simulations = []
    elite_simulations.append(bestSol)
    start = time.process_time()
    bestSol_axu = bestSol.copySol()
    
    
    enter = False
    
    # Main loop for iterations while elapsed time is within the specified maximum time
    while elapsed < t.Maxtime:
        iter += 1
    
        # Generate a new solution using biased-randomized heuristic and perform local search (Tabu Search)
        newSol, cl = heur.constructBRSol_capacity_simulation(sim_learnheuristic(20, var), 0.9)
        newSol = tabuSearch_capacity_simulation(newSol, cl, t.maxIter, heur, sim_learnheuristic(20, var), 0.9)
        whitebox.fit_logistic()
        # Check if the new solution improves the current best solution
        if newSol.of > bestSol.of:
            # Perform the simulation_1 in case of not_penalization_cost is True
            small_simulation.simulation_1(newSol)
    
            # Check if the reliability condition is met, update the best solution, and set the enter flag
            if newSol.reliability["1"] >= 0 and not enter:
                enter = True
                bestSol = newSol.copySol()
                bestSol.time = elapsed
                elite_simulations.append(bestSol)
            # If the enter condition is not met and the new solution has better reliability, update auxiliary best solution
            elif not enter and newSol.mean_stochastic_of["1"] >= bestSol.mean_stochastic_of["1"]:
                bestSol_axu = newSol.copySol()  # Update Solution
                bestSol_axu.time = elapsed
                elite_enter_simulations.append(bestSol_axu)
    
        # Update elapsed time
        elapsed = time.process_time() - start
    
    # Create an instance of the sim_learnheuristic with long simulation time
    large_simulation = sim_learnheuristic(t.long_simulation, var)
 
    
    # Check the enter condition and return the best solution accordingly
    if not enter and len(elite_enter_simulations) > 0:
        for i in elite_enter_simulations:
            large_simulation.simulation_1(i)
        elite_enter_simulations.sort(key=lambda x: x.mean_stochastic_of["1"], reverse=True)
        return elite_enter_simulations[0]
    
    for i in elite_simulations:
        large_simulation.simulation_1(i)
        elite_simulations.sort(key=lambda x: x.mean_stochastic_of["1"], reverse=True)
    
    
    return elite_simulations[0] 






# Function to perform deterministic multistart optimization
def deterministic_multistart(bestSol: Solution, t: Instance) -> Solution:
    var = t.var

    # Initialize weights dictionary with values for weighted heuristic runs
    # we made a dctionary since for each item in that we'll fill the second element by the objective function values
    weights = dict([(i / 10, 0) for i in range(5, 11)]) # {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1.0: 0} 

    # Run the biased-randomized heuristic with different weights to accumulate objective function values
    for _ in range(10):
        for i in weights.items():
            newSol, cl = heur.constructBRSol_capacity_given_weight(i[0])  # Biased-randomized version of the heuristic
            weights[i[0]] += newSol.of
            
    # Sort weights based on accumulated objective function values in descending order
    weights_sort = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
    a = list(weights_sort.items())

    elapsed = 0.0
    start = time.process_time()
    while elapsed < t.Maxtime:
        rand = random.random()
        '''
        The goal is to introduce randomness in selecting weights while avoiding weights exactly equal to 1.
        The use of random.uniform ensures that the weights are chosen within a specified range around the 
        existing values.
        '''
        # Determine the weight for the biased-randomized heuristic based on random values and sorted weights
        if rand < 0.7:
            if a[0][0] != 1:
                weight = random.uniform(a[0][0] - 0.05, a[0][0] + 0.05)
            else:
                weight = random.uniform(a[0][0] - 0.05, a[0][0])
        elif rand < 0.9:
            if a[1][0] != 1:
                weight = random.uniform(a[1][0] - 0.05, a[1][0] + 0.05)
            else:
                weight = random.uniform(a[1][0] - 0.05, a[1][0])
        else:
            if a[2][0] != 1:
                weight = random.uniform(a[2][0] - 0.05, a[2][0] + 0.05)
            else:
                weight = random.uniform(a[2][0] - 0.05, a[2][0])

        # iter += 1
        newSol, cl = heur.constructBRSol_capacity_given_weight(weight)  # Biased-randomized version of the heuristic
        newSol = tabuSearch_capacity(newSol, cl, t.maxIter, heur)  # Local Search (Tabu Search)

        # Check if the new solution improves the BestSol
        if newSol.of > bestSol.of:
            bestSol = newSol.copySol()  # Update Solution
            bestSol.time = elapsed
            # print("New Best Solution:", bestSol.of)

        elapsed = time.process_time() - start
    # print(iter)
    # Create an instance of the sim_learnheuristic with long simulation time
    large_simulation = sim_learnheuristic(t.long_simulation, var)
    
    # Perform simulations on the best solution
    large_simulation.simulation_1(bestSol)
    #large_simulation.simulation_2(bestSol, cl)

    # Return the best solution after deterministic multistart optimization
    return bestSol


'''
Function Main
'''
#open_type = [0,1]
dict_of_types= {}
if __name__ == "__main__":
    tests = readTest("run") # Read the file with the instances to execute
    results_list = [] # Define a list to save the results on that

    for t in tests: # Iterate the list with the instances to execute
        random.seed(t.seed)# Set up the seed to used in the execution
        np.random.seed(t.seed)

        path = "CDP/"+t.instName
        inst = Instance(path) #read instance
        if t.inversa != 0:
            inst.b = sum(inst.capacity) * t.inversa
        alpha = 0
        blackbox = BlackBox()
        whitebox= WhiteBox()
        heur = ConstructiveHeuristic(alpha, t.betaBR, t.betaLS, inst, t.weight, blackbox, whitebox, dict_of_types= dict_of_types)
        bestSol, cl = heur.constructBRSol_capacity()  # Greedy Heurístic

        if t.deterministic:
            bestSol = deterministic_multistart(bestSol, t)
        else:
            if t.not_penalization_cost:
                bestSol = stochastic_multistart_simulation(bestSol, t)
            
            else:
                bestSol = stochastic_multistart(bestSol, t, cl)
                
                
        result_dict = {
            "Instance": t.instName,
            "Deterministic OF": bestSol.of,
            "Objective Function Value in Stochastic Environment": bestSol.mean_stochastic_of["1"],
            "Reliability": bestSol.reliability
        }

        results_list.append(result_dict)
                
        print(t.instName)
        print("of: "+str(bestSol.of))
        print("mean stochastic of: " +str(bestSol.mean_stochastic_of["1"]))
        print("Reliability: "+str(bestSol.reliability))
        
        # Call the plot_wit_some_betas function
        plot_wit_some_betas()
    # Create a DataFrame from the list of results
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to an Excel file
    results_df.to_excel("results.xlsx", index=False)
    
        
        

