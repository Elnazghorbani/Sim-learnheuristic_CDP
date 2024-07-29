

from Solution import *
import random



# Tabu Search Local Search (LS)
def tabuSearch(initSol, cl, maxIter, heur):
    # Make a copy of the initial solution
    sol = initSol.copySol()
    
    # Initialize a counter for consecutive non-improving iterations
    noImprovement = 0
    
    # Continue until the maximum number of iterations is reached
    while noImprovement < maxIter:
        # Select the oldest element from the solution
        n = sol.selected[0]
        
        # Drop the oldest node from the solution
        sol.drop(n)
        
        # Recalculate the candidate list after dropping the node
        heur.recalculateCL(sol, cl, n)
        
        # Partially reconstruct the solution without using the dropped node
        sol = heur.partialReconstruction(sol, cl)
        
        # Reevaluate the objective function value of the solution
        sol.reevaluateSol()
        
        # Insert the dropped node back into the candidate list
        heur.insertNodeToCL(cl, sol, n)
        
        # If the new solution has a higher objective function value, update the initial solution
        if sol.of > initSol.of:
            initSol = sol.copySol()
            noImprovement = 0
        
        # Increment the counter for consecutive non-improving iterations
        noImprovement += 1
    
    # Return the updated initial solution after the Tabu Search process
    return initSol



# Tabu Search Local Search (LS) with Capacity Consideration
def tabuSearch_capacity(initSol, cl, maxIter, heur):
    # Make a copy of the initial solution
    sol = initSol.copySol()
    
    # Initialize a counter for consecutive non-improving iterations
    noImprovement = 0
    
    # Continue until the maximum number of iterations is reached
    while noImprovement < maxIter:
        # Select the oldest element from the solution
        n = sol.selected[0]
        
        # Drop the oldest node from the solution
        sol.drop(n)
        
        # Recalculate the candidate list with capacity after dropping the node
        heur.recalculateCL_capacity(sol, cl, n)
        
        # Partially reconstruct the solution with capacity without using the dropped node
        sol = heur.partialReconstruction_capacity(sol, cl)
        
        # Reevaluate the objective function value of the modified solution
        sol.reevaluateSol()
        
        # Insert the dropped node back into the candidate list with capacity consideration
        heur.insertNodeToCL_capacity(cl, sol, n)
        
        # If the new solution has a higher objective function value, update the initial solution
        if sol.of > initSol.of:
            initSol = sol.copySol()
            noImprovement = 0
        
        # Increment the counter for consecutive non-improving iterations
        noImprovement += 1
    
    # Return the updated initial solution after the Tabu Search process
    return initSol





# Tabu Search with Capacity and Simulation
def tabuSearch_capacity_simulation(initSol, cl, maxIter, heur, simulation, delta):
    # Make a copy of the initial solution
    sol = initSol.copySol()

    # Initialize a counter for consecutive non-improving iterations
    noImprovement = 0

    # Continue until the maximum number of iterations is reached
    while noImprovement < maxIter:
        # Select the oldest element from the solution
        n = sol.selected[0]

        # Drop the oldest node from the solution
        sol.drop(n)

        # Recalculate the candidate list after dropping the node
        heur.recalculateCL_capacity(sol, cl, n)

        # Partially reconstruct the solution using simulation-based adjustment
        sol = heur.partialReconstruction_capacity_simulation(sol, cl, simulation, delta)

        # Reevaluate the objective function value of the solution
        sol.reevaluateSol()

        # Insert the dropped node back into the candidate list
        heur.insertNodeToCL_capacity(cl, sol, n)

        # If the new solution has a higher objective function value, update the initial solution
        if sol.of > initSol.of:
            initSol = sol.copySol()
            noImprovement = 0

        # Increment the counter for consecutive non-improving iterations
        noImprovement += 1

    # Return the updated initial solution after the Tabu Search process
    return initSol
