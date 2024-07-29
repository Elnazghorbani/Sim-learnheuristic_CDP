# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:00:54 2024

@author: eghorbanioskalaei
"""


import math
from Solution import Solution
from objects import Candidate, Candidate_capacity
import random

# Define a class for Constructive Heuristic
class ConstructiveHeuristic:
    def __init__(self, alpha, beta, betaLS, inst, weight, blackbox, whitebox, dict_of_types=None):
        # Initialize parameters
        self.alpha = alpha
        self.firstEdge = 0
        self.beta = beta
        self.betaLS = betaLS
        self.blackbox = blackbox
        self.whitebox = whitebox
        self.instance = inst
        self.weight = weight
        # Dynamic enviroment
        
        self.weather = random.randint(0, 1)
        self.congestion = {i: random.randint(0, 1) for i in range(inst.n)}

        if dict_of_types:
            self.dict_of_types = dict_of_types
        else:
            self.dict_of_types = {i: 1 for i in range(inst.n)}
    # Original Grasp Heuristic
    # Constructive heuristic (Deterministic Version)
    def constructSolution(self):
        # Create a solution object (an empty solution)
        sol = Solution(self.instance)
        
        # Select the first edge from sorted distances
        edge = self.instance.sortedDistances[self.firstEdge]   
        
        # Add vertices of the selected edge to the solution
        sol.add(edge.v1)
        sol.add(edge.v2)
        
        # Update the objective function value based on the selected edge
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        
        # Create a candidate list for remaining vertices
        cl = self.createCL(sol)
        
        # Determine the real alpha value
        realAlpha = self.alpha if self.alpha >= 0 else random.random()
        
        # Iterate until a feasible solution is obtained
        while not sol.isFeasible():
            # Calculate the distance limit for candidate selection
            distanceLimit = cl[0].cost - (realAlpha * cl[len(cl) - 1].cost)
            i = 0
            maxCap = 0
            vWithMaxCap = -1
            
            # Iterate through the candidate list to select the next vertex
            while i < len(cl) and (cl[i].cost >= distanceLimit):
                v = cl[i].v
                vCap = self.instance.capacity[v]
                
                # Update maximum capacity and corresponding vertex
                if vCap > maxCap:
                    maxCap = vCap
                    vWithMaxCap = i
                i += 1
            
            # Remove the selected vertex from the candidate list
            c = cl.pop(vWithMaxCap)
            sol.add(c.v)

            # Update the objective function if the selected vertex improves the solution
            if c.cost < sol.of:
                sol.updateOF(c.v, c.closestV, c.cost)

            # Debug: Uncomment the following lines for debugging
            # if sol.getEvalComplete() != sol.of:
            #     print("MAL: " + str(sol.getEvalComplete()) + " vs " + str(sol.of))

            # Update the candidate list after adding the vertex to the solution
            self.updateCL(sol, cl, c.v)
        
        # Return the final constructed solution
        return sol



    def constructSolution_capacity(self, weight):
        # Initialize a new solution with the given problem instance
        sol = Solution(self.instance)
        
        # Select the first edge from sorted distances
        edge = self.instance.sortedDistances[self.firstEdge]
        
        # Add vertices of the selected edge to the solution
        sol.add(edge.v1)
        sol.add(edge.v2)
        
        # Update the objective function value based on the selected edge
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        
        # Set the weight attribute to the provided weight
        self.weight = weight
        
        # Create a candidate list for remaining vertices considering capacity
        cl = self.createCL_capacity(sol)
        
        # Determine the real alpha value
        realAlpha = self.alpha if self.alpha >= 0 else random.random()
        
        # Iterate until a feasible solution is obtained
        while not sol.isFeasible():
            # Calculate the distance limit for candidate selection
            distanceLimit = cl[0].cost - (realAlpha * cl[len(cl) - 1].cost)
            i = 0
            maxCap = 0
            vWithMaxCap = -1
            
            # Iterate through the candidate list to select the next vertex
            while i < len(cl) and (cl[i].cost >= distanceLimit):
                v = cl[i].v
                vCap = self.instance.capacity[v]
                
                # Update maximum capacity and corresponding vertex
                if vCap > maxCap:
                    maxCap = vCap
                    vWithMaxCap = i
                i += 1
            
            # Remove the selected vertex from the candidate list
            c = cl.pop(vWithMaxCap)
            sol.add(c.v)
    
            # Update the objective function if the selected vertex improves the solution
            if c.dist_min < sol.of:
                sol.updateOF(c.v, c.closestV, c.dist_min)
    
            # Update the candidate list after adding the vertex to the solution
            self.updateCL_capacity(sol, cl, c.v)
        
        # Return the final constructed solution
        return sol


    # BR-Heuristic: Biased Randomized Heuristic
    def constructBRSol(self):
        # Initialize a new solution with the given problem instance
        sol = Solution(self.instance)
    
        # Select a random position in the sorted distances list
        pos = self.getRandomPosition(len(self.instance.sortedDistances), random, self.beta)
        
        # Select the edge corresponding to the random position
        edge = self.instance.sortedDistances[pos]  #in the previous functions, the first edge of the sorted list was chosen, but here we get a random edge of the sorted list
        
        # Add vertices of the selected edge to the solution
        sol.add(edge.v1)
        sol.add(edge.v2)
        
        # Update the objective function value based on the selected edge
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        
        # Create a candidate list for remaining vertices
        cl = self.createCL(sol)
        
        # Iterate until a feasible solution is obtained
        while not sol.isFeasible():
            # Select a random position in the candidate list
            pos = self.getRandomPosition(len(cl), random, self.beta)
            
            # Get the vertex and its corresponding candidate from the random position
            v = cl[pos].v
            c = cl.pop(pos)
            
            # Add the selected vertex to the solution
            sol.add(c.v)
    
            # Update the objective function if the selected vertex improves the solution
            if c.cost < sol.of:
                sol.updateOF(c.v, c.closestV, c.cost)
    
            # Update the candidate list after adding the vertex to the solution
            self.updateCL(sol, cl, c.v)
        
        # Return the final constructed solution and the remaining candidate list
        return sol, cl



    # BR-Heuristic with Capacity Consideration
    def constructBRSol_capacity(self):
        # Initialize a new solution with the given problem instance
        sol = Solution(self.instance)
        
        # Select a random position in the sorted distances list
        pos = self.getRandomPosition(len(self.instance.sortedDistances), random, self.beta)
        
        # Select the edge corresponding to the random position
        edge = self.instance.sortedDistances[pos]
        
        # Add vertices of the selected edge to the solution
        sol.add(edge.v1)
        sol.add(edge.v2)
        
        # Update the objective function value based on the selected edge
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        
        # Generate a random weight between 0.6 and 0.9 for capacity consideration
        self.weight = random.uniform(0.6, 0.9)
        
        # Create a candidate list for remaining vertices considering capacity
        cl = self.createCL_capacity(sol)
        
        # Iterate until a feasible solution is obtained
        while not sol.isFeasible():
            # Select a random position in the candidate list
            pos = self.getRandomPosition(len(cl), random, self.beta)
            
            # Get the vertex and its corresponding candidate from the random position
            v = cl[pos].v
            c = cl.pop(pos)
            
            # Add the selected vertex to the solution
            sol.add(c.v)
            
            # Update the maximum capacity if necessary
            if self.max_capacity < self.instance.capacity[c.v]:
                self.max_capacity = self.instance.capacity[c.v]
            
            # Update the objective function if the selected vertex improves the solution
            if c.dist_min < sol.of:
                sol.updateOF(c.v, c.closestV, c.dist_min)
            
            # Update the candidate list after adding the vertex to the solution
            self.updateCL_capacity(sol, cl, c.v)
        
        # Return the final constructed solution and the remaining candidate list
        return sol, cl



    def constructBRSol_capacity_simulation(self, simulation, delta):
        # Create an empty solution
        sol = Solution(self.instance)
        
        # Select an edge randomly
        pos = self.getRandomPosition(len(self.instance.sortedDistances), random, self.beta)
        edge = self.instance.sortedDistances[pos]
        
        # Add the vertices of the selected edge to the solution
        sol.add(edge.v1)
        sol.add(edge.v2)
        
        # Update the objective function value based on the selected edge
        sol.updateOF(edge.v1, edge.v2, edge.distance)
        
        # Generate a random weight within the specified range
        self.weight = random.uniform(0.6, 0.9)
        
        # Create a candidate list considering capacity
        cl = self.createCL_capacity(sol)
    
        # Perform a fast simulation to get lower and upper bounds
        lower, upper = simulation.fast_simulation(sol)
    
        # Continue adding vertices until the lower bound is greater than or equal to delta
        while lower < delta:
            # Select a vertex randomly from the candidate list
            pos = self.getRandomPosition(len(cl), random, self.beta)
            v = cl[pos].v
            c = cl.pop(pos)
            
            # Add the selected vertex to the solution
            sol.add(c.v)
    
            # Update the maximum capacity if needed
            if self.max_capacity < self.instance.capacity[c.v]:
                self.max_capacity = self.instance.capacity[c.v]
    
            # Update the objective function if the selected vertex improves the solution
            if c.dist_min < sol.of:
                sol.updateOF(c.v, c.closestV, c.dist_min)
    
            # Update the candidate list after adding the vertex to the solution
            self.updateCL_capacity(sol, cl, c.v)
    
            # Perform a fast simulation to get lower and upper bounds for the updated solution
            lower, upper = simulation.fast_simulation(sol)
    
        # Return the final constructed solution and the remaining candidate list
        return sol, cl


    def constructBRSol_capacity_given_weight(self, weight):
        # Create an empty solution
        sol = Solution(self.instance)
    
        # Select a random edge position
        pos = self.getRandomPosition(len(self.instance.sortedDistances), random, self.beta)
        
        # Get the edge from the sorted distances
        edge = self.instance.sortedDistances[pos]
    
        # Add the vertices of the selected edge to the solution
        sol.add(edge.v1)
        sol.add(edge.v2)
    
        # Update the objective function value based on the selected edge
        sol.updateOF(edge.v1, edge.v2, edge.distance)
    
        # Set the weight to the provided value
        self.weight = weight
    
        # Create a candidate list based on capacity
        cl = self.createCL_capacity_det(sol)
    
        # Continue until a feasible solution is obtained
        while not sol.isFeasible():
            # Select a random position in the candidate list
            pos = self.getRandomPosition(len(cl), random, self.beta)
            
            # Get the vertex from the selected position
            v = cl[pos].v
            
            # Remove the selected vertex from the candidate list
            c = cl.pop(pos)
            
            # Add the selected vertex to the solution
            sol.add(c.v)
            
            # Update the maximum capacity if needed
            if self.max_capacity < self.instance.capacity[c.v]:
                self.max_capacity = self.instance.capacity[c.v]
            
            # Update the objective function if the selected vertex improves the solution
            if c.dist_min < sol.of:
                sol.updateOF(c.v, c.closestV, c.dist_min)
    
            # Update the candidate list after adding the vertex to the solution
            self.updateCL_capacity_det(sol, cl, c.v)
    
        # Return the final solution and the remaining candidate list
        return sol, cl


    # Function to construct a solution with capacity consideration using an adjusted weight range
    def constructBRSol_capacity_ajusted(self, ajusted):
        # Create an empty solution
        sol = Solution(self.instance)
        
        # Select a random edge position
        pos = self.getRandomPosition(len(self.instance.sortedDistances), random, self.beta)
        
        # Get the edge from the sorted distances
        edge = self.instance.sortedDistances[pos]
        
        # Add the vertices of the selected edge to the solution
        sol.add(edge.v1)
        sol.add(edge.v2)
        
        # Update the objective function value based on the selected edge
        sol.updateOF(edge.v1, edge.v2, edge.distance)
    
        # Generate a random number and select a weight range based on adjusted probabilities
        random_number = random.random()
        before = 0
        selected = list(ajusted.keys())[-1]
        for i in ajusted.items():
            if random_number <= i[1] + before:
                selected = i[0]
                break
            else:
                before += i[1]
    
        # Set the weight to a random value within the selected range
        self.weight = random.uniform(selected[0], selected[1])
    
        # Create a candidate list for remaining vertices considering capacity
        cl = self.createCL_capacity(sol)
    
        # Continue until a feasible solution is obtained
        while not sol.isFeasible():
            # Select a random position in the candidate list
            pos = self.getRandomPosition(len(cl), random, self.beta)
            
            # Get the vertex and its corresponding candidate from the random position
            v = cl[pos].v
            c = cl.pop(pos)
            
            # Add the selected vertex to the solution
            sol.add(c.v)
            
            # Update the maximum capacity if necessary
            if self.max_capacity < self.instance.capacity[c.v]:
                self.max_capacity = self.instance.capacity[c.v]
            
            # Update the objective function if the selected vertex improves the solution
            if c.dist_min < sol.of:
                sol.updateOF(c.v, c.closestV, c.dist_min)
            
            # Update the candidate list after adding the vertex to the solution
            self.updateCL_capacity(sol, cl, c.v)
    
        # Return the final solution, remaining candidate list, and the selected weight range
        return sol, cl, selected





    # Function to create a candidate list without considering capacity
    def createCL(self, sol):
        # Get the problem instance from the solution
        instance = sol.instance
        
        # Get the number of vertices in the instance
        n = instance.n
        
        # Initialize an empty candidate list
        cl = []  # Candidate List of nodes
        
        # Iterate through all vertices in the instance
        for v in range(0, n):
            # Skip vertices that are already selected in the solution
            if v in sol.selected:
                continue
            
            # Calculate the distance and closest vertex for the current vertex
            vMin, minDist = sol.distanceTo(v)
            
            # Create a Candidate object for the current vertex and add it to the list
            c = Candidate(v, vMin, minDist)
            cl.append(c)
        
        # Sort the candidate list based on the distance in descending order
        cl.sort(key=lambda x: x.cost, reverse=True) #the cost attribute represents the distance of a candidate vertex to its closest selected vertex in the solution
        
        # Return the created candidate list
        return cl
    
    def createCL_capacity(self, sol):
        # Get the problem instance from the solution
        instance = sol.instance
        
        # Get the number of vertices in the instance
        n = instance.n
        
        # Initialize an empty candidate list
        cl = []  # Candidate List of nodes
        
        # Separate lists to store nodes and their capacities
        nodes = []
        capacity = []
        
        # Iterate through all vertices in the instance
        for v in range(0, n):
            # Skip vertices that are already selected in the solution
            if v in sol.selected:
                # Store the capacity of selected vertices
                capacity.append(instance.capacity[v])
                continue
            
            # Calculate the distance and closest vertex for the current vertex
            vMin, minDist = sol.distanceTo(v)
            
            # Create a Candidate object for the current vertex and add it to the nodes list
            c = Candidate(v, vMin, minDist)
            nodes.append(c)
    
        # Calculate the maximum distance and capacity in the nodes and capacity lists (calculate the normalization factors)
        self.max_min_dist = max([x.cost for x in nodes])
        self.max_capacity = max(capacity)
    
        # Iterate through the nodes to calculate the cost and create Candidate_capacity objects
        #The normalization allows the algorithm to balance the importance of distances and capacities in the candidate 
        #selection process.
        for i in nodes:
            #open_type_node = open_type.copy()
            #open_type_node[self.dict_of_types[i.v]] += 1/self.instance.n
            simulate = { "weather": self.weather, "congestion": self.congestion[i.v]}

            cost = i.cost / self.max_min_dist * self.weight + instance.capacity[i.v] / self.max_capacity * \
                   (1 - self.weight) * self.whitebox.get_value_with_dict(instance.capacity[i.v], simulate)

            c = Candidate_capacity(i.v, i.closestV, i.cost, cost)
            c.modified_capacity = instance.capacity[i.v] * self.whitebox.get_value_with_dict(instance.capacity[i.v], simulate)
            cl.append(c)

        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos

        return cl
            



    def createCL_capacity_det(self, sol):
        # Get the problem instance from the solution
        instance = sol.instance
        
        # Get the number of vertices in the instance
        n = instance.n
        
        # Initialize an empty candidate list
        cl = []  # Candidate List of nodes
        
        # Separate lists to store nodes and their capacities
        nodes = []
        capacity = []
        
        # Iterate through all vertices in the instance
        for v in range(0, n):
            # Skip vertices that are already selected in the solution
            if v in sol.selected:
                # Store the capacity of selected vertices
                capacity.append(instance.capacity[v])
                continue
            
            # Calculate the distance and closest vertex for the current vertex
            vMin, minDist = sol.distanceTo(v)
            
            # Create a Candidate object for the current vertex and add it to the nodes list
            c = Candidate(v, vMin, minDist)
            nodes.append(c)
    
        # Calculate the maximum distance and capacity in the nodes and capacity lists (calculate the normalization factors)
        self.max_min_dist = max([x.cost for x in nodes])
        self.max_capacity = max(capacity)
    
        # Iterate through the nodes to calculate the cost and create Candidate_capacity objects
        #The normalization allows the algorithm to balance the importance of distances and capacities in the candidate 
        #selection process.
        for i in nodes:
            cost = i.cost / self.max_min_dist * self.weight + instance.capacity[i.v] / self.max_capacity * (1 - self.weight)
            c = Candidate_capacity(i.v, i.closestV, i.cost, cost)
            cl.append(c)
        
        # Sort the candidate list based on the cost in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)
        
        # Return the created candidate list
        return cl         
            
            
        #     cost = i.cost / self.max_min_dist * self.weight + instance.capacity[i.v] / self.max_capacity * (1 - self.weight)
        #     c = Candidate_capacity(i.v, i.closestV, i.cost, cost)
        #     cl.append(c)
        
        # # Sort the candidate list based on the cost in descending order
        # cl.sort(key=lambda x: x.cost, reverse=True)
        
        # # Return the created candidate list
        # return cl





    def update_cl(self, sol, cl, last_added):
        instance = sol.instance
        for c in cl:
            d_to_last = instance.distance[last_added][c.v]
            if d_to_last < c.cost:
                c.cost = d_to_last
                c.closest_v = last_added
        cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos







    # Function to update the candidate list without considering capacity
    def updateCL(self, sol, cl, lastAdded):
        # Get the problem instance from the solution
        instance = sol.instance
        
        # Update the cost and closestV attributes for each candidate in the list
        for c in cl:
            dToLast = instance.distance[lastAdded][c.v]
            
            # If the distance to the last added vertex is smaller, update cost and closestV
            if dToLast < c.cost:
                c.cost = dToLast
                c.closestV = lastAdded
        
        # Sort the candidate list based on the updated cost in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)




    # Function to update the candidate list considering capacity
    def updateCL_capacity(self, sol, cl, lastAdded):
        
        # Get the problem instance from the solution
        instance = sol.instance
        
        # Update the dist_min and closestV attributes for each candidate in the list
        for c in cl:
            dToLast = instance.distance[lastAdded][c.v]
            
            # If the distance to the last added vertex is smaller, update dist_min and closestV
            if dToLast < c.dist_min:
                c.dist_min = dToLast
                c.closestV = lastAdded
        
        # Update the maximum dist_min value in the candidate list
        self.max_min_dist = max([x.dist_min for x in cl])
        
        # Update the cost attribute for each candidate considering weight and capacity
        for c in cl:
            #open_type_node = open_type.copy()
            #open_type_node[self.dict_of_types[c.v]] += 1/self.instance.n
            simulate = { "weather": self.weather, "congestion": self.congestion[c.v]}
            # If the max_min_dist is not zero, calculate the cost with weight and capacity
            if self.max_min_dist != 0:
                c.cost = c.dist_min / self.max_min_dist * self.weight + instance.capacity[c.v] / self.max_capacity * (1 - self.weight)* self.whitebox.get_value_with_dict(instance.capacity[c.v], simulate)
            # If max_min_dist is zero, calculate the cost based on capacity only
                c.modified_capacity = instance.capacity[c.v] * self.whitebox.get_value_with_dict(instance.capacity[c.v], simulate)
            else:
                c.cost = instance.capacity[c.v] / self.max_capacity * (1 - self.weight)* self.whitebox.get_value_with_dict(instance.capacity[c.v], simulate)
                c.modified_capacity = instance.capacity[c.v] #* self.whitebox.get_value_with_dict(simulate)
                
        # Sort the candidate list based on the updated cost in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)
        




    # Function to update the candidate list considering capacity
    def updateCL_capacity_det(self, sol, cl, lastAdded):
        # Get the problem instance from the solution
        instance = sol.instance
        
        # Update the dist_min and closestV attributes for each candidate in the list
        for c in cl:
            dToLast = instance.distance[lastAdded][c.v]
            
            # If the distance to the last added vertex is smaller, update dist_min and closestV
            if dToLast < c.dist_min:
                c.dist_min = dToLast
                c.closestV = lastAdded
        
        # Update the maximum dist_min value in the candidate list
        self.max_min_dist = max([x.dist_min for x in cl])
        
        # Update the cost attribute for each candidate considering weight and capacity
        for c in cl:
            # If the max_min_dist is not zero, calculate the cost with weight and capacity
            if self.max_min_dist != 0:
                c.cost = c.dist_min / self.max_min_dist * self.weight + instance.capacity[c.v] / self.max_capacity * (1 - self.weight)
            # If max_min_dist is zero, calculate the cost based on capacity only
            else:
                c.cost = instance.capacity[c.v] / self.max_capacity * (1 - self.weight)
        
        # Sort the candidate list based on the updated cost in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)
            



    '''
    Obtain a random number between 0 and the maximum size of a list using a Geometric distribution behavior.
    parameters:
    Size: Size of a list
    beta: Beta parameter of the Geometric distribution
    '''
    # why we used geometry distribution? why we didn't pick the number randomly? because we want a biased selection. but why we don't select the first element of the list instead of geometry distribution?
    #A smaller beta value might lead to more exploration (selecting positions randomly), while a larger beta value could bias the selection towards positions with higher probabilities.

    #p(x=k)= p.(1-p)^k-1  beta is the parameter of the geometry distribution (p), and 'size' is the size of the set 
    #that we're gonna chose a number in that
    # Function to get a random position with bias
    def getRandomPosition(self, size, random, beta): 
        # Calculate a random floating-point number between 0 and 1
        rand_num = random.random()
    
        # Calculate the index using a biased logarithmic transformation
        index = int(math.log(rand_num) / math.log(1 - beta))
        
        # Ensure the index is within the valid range (0 to size-1)
        index = index % size
        
        # Return the calculated index
        return index


    # This function is responsible for recalculating the cost and closest vertex of each candidate in the 
    # candidate list (cl) after dropping a node (lastDrop) from the solution (sol)
    def recalculateCL(self, sol, cl, lastDrop):
        # Get the problem instance from the solution
        instance = sol.instance
        
        # Iterate through the candidate list
        for c in cl:
            # Check if the last dropped node was the closest node to the current candidate
            if lastDrop == c.closestV:
                # Calculate the distance and closest vertex for the current candidate
                vMin, minDist = sol.distanceTo(c.v)
                
                # Update the cost and closest vertex of the candidate based on the new information
                c.cost = minDist
                c.closestV = vMin
        
        # Sort the candidate list based on the updated distance in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)





    # Recalculate the candidate list with capacity after dropping a node
    def recalculateCL_capacity(self, sol, cl, lastDrop):
        # Get the problem instance from the solution
        instance = sol.instance
        
        # Iterate through the candidate list
        for c in cl:
            # Check if the last dropped node was the closest node to the current candidate
            if lastDrop == c.closestV:
                # Update the candidate's minimum distance and closest vertex
                vMin, minDist = sol.distanceTo(c.v)
                c.dist_min = minDist
                c.closestV = vMin
    
        # Update the maximum minimum distance in the candidate list
        self.max_min_dist = max([x.dist_min for x in cl])
    
        # Check if the last dropped node had the maximum capacity in the solution
        if self.max_capacity == instance.capacity[lastDrop]:
            # Update the maximum capacity considering the remaining selected nodes
            self.max_capacity = max([instance.capacity[i] for i in sol.selected])
        # we need two paprameters: max_min_dist and max_capacity. After building these two parameters, we use a normalization 
        # formulation by giving weight to both parameters and get the cost as our final result.
        # Update the cost of each candidate based on distance and capacity considerations
        for c in cl:
            # Calculate the cost with capacity considerations
            if self.max_min_dist != 0:
                c.cost = c.dist_min / self.max_min_dist * self.weight + instance.capacity[c.v] / self.max_capacity * (
                    1 - self.weight)
            else:
                # If max_min_dist is zero, only consider capacity in the cost calculation
                c.cost = instance.capacity[c.v] / self.max_capacity * (1 - self.weight)
    
        # Sort the candidate list based on cost in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)





    # Used in the local search to include new nodes in the solution
    def partialReconstruction(self, sol, cl):
        # Continue until the solution is feasible
        while (not sol.isFeasible()):
            # Select a random position in the candidate list
            pos = self.getRandomPosition(len(cl), random, self.betaLS)
            
            # Get the candidate from the selected position and remove it from the candidate list
            c = cl.pop(pos)
            
            # Add the selected vertex to the solution
            sol.add(c.v)
            
            # Update the candidate list after adding the vertex to the solution
            self.updateCL(sol, cl, c.v)
        
        # Return the partially reconstructed solution
        return sol





    def partialReconstruction_capacity(self, sol, cl):
        # While the solution is not feasible:
        while (not sol.isFeasible()):
            # Select a random position in the candidate list
            pos = self.getRandomPosition(len(cl), random, self.betaLS)
            # Retrieve and remove the candidate at the selected position from the candidate list
            c = cl.pop(pos)
            # Add the vertex associated with the selected candidate to the solution
            sol.add(c.v)
            # Update the candidate list considering the changes made to the solution
            self.updateCL_capacity(sol, cl, c.v)
        # Return the modified solution after the partial reconstruction process
        return sol


    
    # Used in the LS to include new nodes in the solution with capacity and simulation considerations
    def partialReconstruction_capacity_simulation(self, sol, cl, simulation, delta):
        # Perform a fast simulation to get lower and upper bounds
        lower, upper = simulation.fast_simulation(sol)
    
        # Continue adding nodes until the lower bound exceeds the specified delta
        while lower < delta:
            # Select a random position in the candidate list
            pos = self.getRandomPosition(len(cl), random, self.betaLS)
    
            # Remove the candidate from the candidate list
            c = cl.pop(pos)
    
            # Add the selected node to the solution
            sol.add(c.v)
    
            # Update the candidate list after adding the node to the solution
            self.updateCL_capacity(sol, cl, c.v)
    
            # Perform a fast simulation to update lower and upper bounds
            lower, upper = simulation.fast_simulation(sol)
    
        # Return the partially reconstructed solution with capacity and simulation considerations
        return sol





    # This function choose the closest node in the sol to the v (which is in cl list), then add this node to cl
    def insertNodeToCL(self, cl, sol, v):
        # Calculate the distance and nearest selected vertex for the new node
        vMin, minDist = sol.distanceTo(v)
        
        # Create a Candidate object for the new node and add it to the candidate list
        c = Candidate(v, vMin, minDist)
        cl.append(c)
        
        # Sort the candidate list based on the distance in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)
        
        return cl





    def insertNodeToCL_capacity(self, cl, sol, v):
        # Get the capacity of the selected vertex
        capacity_v = self.instance.capacity[v]
        
        # Update the maximum capacity if needed
        if self.max_capacity < capacity_v:
            self.max_capacity = capacity_v
    
        # Calculate the distance and closest vertex for the selected vertex in the solution
        vMin, minDist = sol.distanceTo(v)
    
        # Update the maximum minimum distance if needed
        if self.max_min_dist < minDist:
            self.max_min_dist = minDist
    
        # Calculate the cost considering distance and capacity with weight consideration
        if self.max_min_dist != 0:
            cost = minDist / self.max_min_dist * self.weight + capacity_v / self.max_capacity * (1 - self.weight)
        else:
            # If max_min_dist is zero, only consider capacity in the cost calculation
            cost = capacity_v / self.max_capacity * (1 - self.weight)
    
        # Create a Candidate_capacity object with the calculated cost
        c = Candidate_capacity(v, vMin, minDist, cost)
    
        # Add the candidate to the candidate list and sort it in descending order based on cost
        cl.append(c)
        cl.sort(key=lambda x: x.cost, reverse=True)
    
        # Return the updated candidate list
        return cl
