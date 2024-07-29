
import random  
import numpy as np  
from ConstructiveHeuristic import *   

# Generates a dynamic capacity using a machine learning (regression) model 
def getDynamicValue(node, solution, operational_dis, seasonal_demand):
    # A multiple regression model that returns the standard capacity of nodes if seasonal demand and 
    # operation disruption levels are 0, while increasing the cost as they approach to 1
    # It is designed so the dynamic capacity varies between the capacity of the candidate and 1.5 * capacity of the candidate
    
    # Get the standard capacity of the node from the solution instance
    cap = solution.instance.capacity[node]
    
    # Regression model coefficients
    b0 = 0  # Independent term in a regression model
    b_e = 1  # Coefficient for node standard capacity
    b_w = 0.2 * cap  # Coefficient for seasonal demand (less influential factor)
    b_t = 0.3 * cap  # Coefficient for operation disruption (more influential factor)
    
    # Calculate dynamic capacity using the regression model
    dynamicCapacity = b0 + b_e * cap + b_w * operational_dis + b_t * seasonal_demand
    
    return dynamicCapacity              

# Define a class named sim_learnheuristic
class sim_learnheuristic:
    # Constructor with parameters simulations and variance
    def __init__(self, simulations, variance):
        # Initialize the sim_learnheuristic object with simulations and variance
        self.simulations = simulations
        self.var = variance

    # Method to generate a random value based on a given mean using lognormal distribution
    def random_value(self, mean):
        # Generate a random value from a lognormal distribution with log(mean) as the mean and self.var as the variance
        return np.random.lognormal(np.log(mean), self.var)

    # Method definition for simulation_1 in the sim_learnheuristic class
    def simulation_1(self, solution: Solution):
        # Initialize fail count and an empty list for capacity values
        fail = 0
        capacity = []
        
        # Set the stochastic objective function for scenario 1 to the current objective function value
        solution.stochastic_of["1"] = solution.of # Optimize
        operational_dis = random.random()  # Generate a random operational disruption level
        
        # Loop through the number of simulations
        for _ in range(self.simulations):
            # Initialize stochastic_capacity to zero for each simulation
            stochastic_capacity = 0
    
            # Loop through selected nodes in the current solution
            for node in solution.selected:
                # Generate a random seasonal demand adversity level
                seasonal_demand = random.random()
                
                # Generate a dynamic capacity using the regression model
                capacity_node = getDynamicValue(node, solution, operational_dis, seasonal_demand)
                
                # Accumulate stochastic capacities
                stochastic_capacity += capacity_node 
    
            # Check if the total stochastic capacity is less than the capacity constraint
            if stochastic_capacity < solution.instance.b:
                # Increment fail count if the constraint is not satisfied
                fail += 1
    
            # Append the stochastic_capacity value to the capacity list
            capacity.append(stochastic_capacity)
    
        # Calculate reliability for scenario 1
        solution.reliability["1"] = (self.simulations - fail) / self.simulations
        # Calculate the mean of stochastic capacities and store it in total_stochastic_capacity
        solution.total_stochastic_capacity["1"] = np.mean(stochastic_capacity)

    # Method definition for simulation_2 in the sim_learnheuristic class
    def simulation_2(self, solution: Solution, cl):
        # Initialize fail count and an empty list for capacity values
        fail = 0
        capacity = []
    
        # Initialize stochastic_of["2"] as a list in the solution object
        solution.stochastic_of["2"] = []
    
        # Generate a random operational disruption level
        operational_dis = random.random()
        
        # Loop through the number of simulations
        for _ in range(self.simulations):
            # Create an auxiliary solution object with the same instance as the original solution
            aux_solution = Solution(solution.instance)
            # Set auxiliary solution's objective function values to those of the original solution
            aux_solution.of = solution.of
            aux_solution.vMin1 = solution.vMin1
            aux_solution.vMin2 = solution.vMin2
            of = solution.of
            # Initialize stochastic_capacity to zero for each simulation
            stochastic_capacity = 0
    
            # Loop through selected nodes in the current solution
            for node in solution.selected:
                # Generate a random seasonal demand adversity level
                seasonal_demand = random.random()
                
                # Generate a dynamic capacity using the regression model
                capacity_node = getDynamicValue(node, solution, operational_dis, seasonal_demand)
                
                # Accumulate stochastic capacities
                stochastic_capacity += capacity_node 
    
            # Check if the total stochastic capacity is less than the capacity constraint
            if stochastic_capacity < solution.instance.b:
                # Increment fail count if the constraint is not satisfied
                fail += 1
    
                # Initialize an index variable for iterating through candidate list (cl)
                i = 0
                '''
               "In this section of the code, if the total stochastic capacity does not satisfy the demand, a repair 
               operation is initiated. During this operation, new nodes from the CL list are added to the solution, 
               and their deterministic capacity is also included in the total stochastic capacity until the demand 
               is met. The question arises: wouldn't it be better to add the stochastic capacities of the new nodes
               instead of their deterministic counterparts? Additionally, this part suggests that if the demand 
               constraint is not satisfied, set the total stochastic capacities equal to the total deterministic 
               capacities. However, the purpose of this loop is unclear."
                '''
                # Perform repair operation until the constraint is satisfied. 
                while stochastic_capacity < solution.instance.b:
                    # Get the vertex from the candidate list
                    v = cl[i].v
                    # Update stochastic_capacity by adding the capacity of the selected vertex
                    stochastic_capacity += solution.instance.capacity[v]
    
                    # If the objective function value of the current candidate is better than the current of value
                    if of > cl[i].dist_min:
                        # Update the objective function values in the auxiliary solution
                        aux_solution.updateOF(cl[i].v, cl[i].closestV, cl[i].dist_min)
    
                    # Update candidate list and capacities in the auxiliary solution
                    self.updateCL_capacity(aux_solution, cl, cl[i].v)
                    # Move to the next candidate
                    i += 1
    
                # Reset stochastic_capacity to the sum of capacities in the original solution
                stochastic_capacity = sum(solution.instance.capacity)
    
            # Append the stochastic_capacity value to the capacity list
            capacity.append(stochastic_capacity)
            # Append the objective function value to the stochastic_of["2"] list in the solution
            solution.stochastic_of["2"].append(of)
    
        # Calculate reliability for scenario 2
        solution.reliability["2"] = (self.simulations - fail) / self.simulations
        # Calculate the mean of stochastic capacities and store it in total_stochastic_capacity["2"]
        solution.total_stochastic_capacity["2"] = np.mean(capacity)
        # Calculate the mean of objective function values for scenario 2
        solution.mean_stochastic_of["2"] = np.mean(solution.stochastic_of["2"])

    # Method definition for fast_simulation in the sim_learnheuristic class
    def fast_simulation(self, solution: Solution):
        # Initialize the failure count
        fail = 0
        
        # Generate a random operational disruption level
        operational_dis = random.random()
        
        # Loop through the number of simulations
        for _ in range(self.simulations):
            # Initialize stochastic_capacity to zero for each simulation
            stochastic_capacity = 0
    
            # Loop through selected nodes in the current solution
            for node in solution.selected:
                # Generate a random seasonal demand adversity level
                seasonal_demand = random.random()
                
                # Generate a dynamic capacity using the regression model
                capacity_node = getDynamicValue(node, solution, operational_dis, seasonal_demand)
                
                # Accumulate stochastic capacities
                stochastic_capacity += capacity_node
    
            # Check if the total stochastic capacity is less than the capacity constraint
            if stochastic_capacity < solution.instance.b:
                # Increment fail count if the constraint is not satisfied
                fail += 1
        
        # Calculate the probability, variance, confidence interval lower and upper bounds, and return them
        p = (self.simulations - fail) / self.simulations
        variance = ((p * (1 - p)) / self.simulations) ** (1/2)
        inf = p - 1.96 * variance
        sup = p + 1.96 * self.var
        return (inf, sup)

    # Method definition for updating capacities in the candidate list
    def updateCL_capacity(self, sol, cl, lastAdded):
        # Get the instance from the solution
        instance = sol.instance
        
        # Loop through candidates in the candidate list
        for c in cl:
            # Calculate the distance to the last added vertex
            dToLast = instance.distance[lastAdded][c.v]
            
            # Update the minimum distance and closest vertex in the candidate list if needed
            if dToLast < c.dist_min:
                c.dist_min = dToLast
                c.closestV = lastAdded

        # Calculate the maximum minimum distance in the candidate list
        self.max_min_dist = max([x.dist_min for x in cl])
        
        # Loop through candidates in the candidate list
        for c in cl:
            # Update the cost of each candidate based on distance and capacity
            if self.max_min_dist != 0:
                c.cost = c.dist_min / self.max_min_dist * 0.8 + instance.capacity[c.v] / max(instance.capacity) * (1 - 0.8)
            else:
                c.cost = instance.capacity[c.v] / max(instance.capacity) * (1 - 0.8)
        
        # Sort the candidate list based on cost in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)  # Order distance from largest to smallest






