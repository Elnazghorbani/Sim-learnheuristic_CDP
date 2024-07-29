import math
import copy
import random
import numpy as np
from ConstructiveHeuristic import *
from scipy.stats import truncnorm
from objects import *



def setCapType(solution):
    dic_type = {}
    for node in solution.selected:
        if node % 2 == 0:
            dic_type[node] = 0  
        else:
            dic_type[node] = 1 
    return dic_type



class sim_learnheuristic:

    def __init__(self, simulations, variance):
        self.simulations = simulations
        self.var = variance
        self.blackbox = BlackBox()  # Initialize BlackBox instance
        self.whitebox = WhiteBox()  # Initialize WhiteBox instance 

       
    def random_value(self, mean):
        variance = mean * self.var
        mu = math.log(mean**2 / math.sqrt(variance + mean**2))
        sigma = math.sqrt(math.log(1 + variance / mean**2))
        return np.random.lognormal(mean=mu, sigma=sigma)

    def simulation_1(self, solution: Solution):
        fail = 0
        dic_type = setCapType(solution)
        capacity = []
        solution.stochastic_of["1"] = solution.of

        for _ in range(self.simulations):
            stochastic_capacity = 0

            for node in solution.selected:
                
                if dic_type[node] == 0: 
                    capacity_node = self.random_value(solution.instance.capacity[node])  
                elif dic_type[node] == 1:
                    
                    cap_node = solution.instance.capacity[node]
                    weather = solution.weather
                    congestion = solution.congestion[node]
                    probValue = self.blackbox.simulate(cap_node=cap_node, weather=weather, congestion=congestion, verbose=False)
                    capacity_node =   probValue* solution.instance.capacity[node]  
                    self.whitebox.add_data(cap_node=cap_node, weather=weather, congestion=congestion, variable=capacity_node)
                    
                
                stochastic_capacity += capacity_node
            if stochastic_capacity < solution.instance.b:
                fail += 1
            capacity.append(stochastic_capacity)
            
        
        solution.mean_stochastic_of["1"] = ((self.simulations - fail) * solution.of) / self.simulations
        solution.reliability["1"] = (self.simulations - fail) / self.simulations
        solution.total_stochastic_capacity["1"] = np.mean(capacity)



 
    def simulation_2(self, solution: Solution, cl):
        fail = 0
        dic_type = setCapType(solution)
        capacity = []
        solution.stochastic_of["2"] = []
       
        
        for _ in range(self.simulations):
            aux_solution = Solution(solution.instance)
            aux_solution.of = solution.of
            aux_solution.of = solution.vMin1
            aux_solution.of = solution.vMin2
            of = solution.of
            stochastic_capacity = 0
            for node in solution.selected:
                  if dic_type[node] == 0: 
                    capacity_node = self.random_value(solution.instance.capacity[node])  
                elif dic_type[node] == 1:
                    
                    cap_node = solution.instance.capacity[node]
                    weather = solution.weather
                    congestion = solution.congestion[node]
                    probValue = self.blackbox.simulate(cap_node=cap_node, weather=weather, congestion=congestion, verbose=False)
                    capacity_node =   probValue* solution.instance.capacity[node]  
                    self.whitebox.add_data(cap_node=cap_node, weather=weather, congestion=congestion, variable=capacity_node)
                  
                stochastic_capacity += capacity_node
            
            if stochastic_capacity < solution.instance.b:
                fail += 1
                solution.of = 0
                break  # Exit the loop immediately if capacity is insufficient
                i = 0
                while stochastic_capacity < solution.instance.b:
                    v = cl[i].v
                    stochastic_capacity += solution.instance.capacity[v]
                    if of > cl[i].dist_min:
                        of = cl[i].dist_min
                    aux_solution.updateOF(cl[i].v, cl[i].closestV, cl[i].dist_min)
                    self.updateCL_capacity(aux_solution, cl, cl[i].v)
                    i += 1
                   
                stochastic_capacity = sum(solution.instance.capacity)

            
            
            capacity.append(stochastic_capacity)
            solution.stochastic_of["2"].append(of)
        
        solution.reliability["2"] = (self.simulations - fail) / self.simulations
        solution.total_stochastic_capacity["2"] = np.mean(capacity)
        solution.mean_stochastic_of["2"] = np.mean(solution.stochastic_of["2"])
        

    def fast_simulation(self, solution: Solution):
        fail = 0
        dic_type = setCapType(solution)
       
        for _ in range(self.simulations):
            stochastic_capacity = 0
            for node in solution.selected:
                  if dic_type[node] == 0: 
                    capacity_node = self.random_value(solution.instance.capacity[node])  
                elif dic_type[node] == 1:
                    
                    cap_node = solution.instance.capacity[node]
                    weather = solution.weather
                    congestion = solution.congestion[node]
                    probValue = self.blackbox.simulate(cap_node=cap_node, weather=weather, congestion=congestion, verbose=False)
                    capacity_node =   probValue* solution.instance.capacity[node]  
                    self.whitebox.add_data(cap_node=cap_node, weather=weather, congestion=congestion, variable=capacity_node)
                    
                    
                stochastic_capacity += capacity_node
            
            if stochastic_capacity < solution.instance.b:
                fail += 1
                solution.of = 0
                break  # Exit the loop immediately if capacity is insufficient
            

            
        p = (self.simulations - fail) / self.simulations
        variance = ((p*(1-p))/self.simulations)**(1/2)
        inf = p-1.96*variance
        sup = p+1.96*self.var
        
        return (inf, sup)



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
            simulate = { "weather": self.weather, "congestion": self.congestion[c.v]}
            # If the max_min_dist is not zero, calculate the cost with weight and capacity
            if self.max_min_dist != 0:
                c.cost = c.dist_min / self.max_min_dist * self.weight + instance.capacity[c.v] / self.max_capacity * (1 - self.weight)* self.whitebox.get_value_with_dict(cap_node, simulate)
            # If max_min_dist is zero, calculate the cost based on capacity only
                c.modified_capacity = instance.capacity[c.v] * self.whitebox.get_value_with_dict(cap_node, simulate)
            else:
                c.cost = instance.capacity[c.v] / self.max_capacity * (1 - self.weight)* self.whitebox.get_value_with_dict(cap_node, simulate)
                c.modified_capacity = instance.capacity[c.v] #* self.whitebox.get_value_with_dict(simulate)
                
        # Sort the candidate list based on the updated cost in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)
        
