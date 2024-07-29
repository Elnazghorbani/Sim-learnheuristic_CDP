#******stochastic/ dynamic solution



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
                
                # #capacity_node = self.random_value(solution.instance.capacity[node]) #solution.instance.capacity[node]
                # cap_node = solution.instance.capacity[node]
                # weather = solution.weather
                # congestion = solution.congestion[node]
                # probValue = self.blackbox.simulate(cap_node=cap_node, weather=weather, congestion=congestion, verbose=False)
                # capacity_node =   probValue* solution.instance.capacity[node]  #  
                # #print("capacity_node:",capacity_node)
                # # Record data into WhiteBox
                # self.whitebox.add_data(cap_node=cap_node, weather=weather, congestion=congestion, variable=capacity_node)
                
                if dic_type[node] == 0: 
                    capacity_node = self.random_value(solution.instance.capacity[node])   #solution.instance.capacity[node]
                # #     #print("deterministic cap:",solution.instance.capacity[node], "stochastic cap:", capacity_node)
                elif dic_type[node] == 1:
                    
                    cap_node = solution.instance.capacity[node]
                    weather = solution.weather
                    congestion = solution.congestion[node]
                    probValue = self.blackbox.simulate(cap_node=cap_node, weather=weather, congestion=congestion, verbose=False)
                    capacity_node =   probValue* solution.instance.capacity[node]  #  
                    #print("capacity_node:",capacity_node)
                #     # Record data into WhiteBox
                    self.whitebox.add_data(cap_node=cap_node, weather=weather, congestion=congestion, variable=capacity_node)
                    
                    

                    
                    # Use white box to predict capacity
                    #probValue2 = self.whitebox.simulate(weather=weather, congestion=congestion, verbose=False)
                    #capacity_node2 = probValue2* solution.instance.capacity[node]
                    # if capacity_node != solution.instance.capacity[node] and capacity_node != 0:
                    #     print('true')
                    # else:
                    #     print('false')
                    #print("deterministic cap:",solution.instance.capacity[node], "dynamic cap:", capacity_node)
                
                stochastic_capacity += capacity_node
                #print("final cap:", stochastic_capacity)
            if stochastic_capacity < solution.instance.b:
                fail += 1
            capacity.append(stochastic_capacity)
            
        #self.whitebox.fit_with_probability(self.whitebox.fit_logistic, prob='zero_at_t_max_decreasing_probability', max_time=self.simulations, time=_ + 1)
        #print(fail)    
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
                
                #capacity_node = self.random_value(solution.instance.capacity[node]) #solution.instance.capacity[node]
                cap_node = solution.instance.capacity[node] 
                weather = solution.weather
                congestion = solution.congestion[node]
                probValue = self.blackbox.simulate(cap_node = cap_node, weather=weather, congestion=congestion, verbose=False)
                capacity_node =   probValue* solution.instance.capacity[node] #  
            #     #print("capacity_node:",capacity_node)
            #     # Record data into WhiteBox
                self.whitebox.add_data(cap_node =cap_node, weather=weather, congestion=congestion, variable=capacity_node)
                
                
                # if dic_type[node] == 0:
                    
                #     capacity_node = self.random_value(solution.instance.capacity[node])  #solution.instance.capacity[node]

                # elif dic_type[node] == 1:
                    
                #     cap_node = solution.instance.capacity[node] 
                #     weather = solution.weather
                #     congestion = solution.congestion[node]
                #     probValue = self.blackbox.simulate(cap_node = cap_node, weather=weather, congestion=congestion, verbose=False)
                #     capacity_node =   probValue* solution.instance.capacity[node] #  
                # #     #print("capacity_node:",capacity_node)
                # #     # Record data into WhiteBox
                #     self.whitebox.add_data(cap_node =cap_node, weather=weather, congestion=congestion, variable=capacity_node)
                    

                    
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
                        #sol.updateOF(c.v, c.closestV, c.dist_min)
                        of = cl[i].dist_min
                    aux_solution.updateOF(cl[i].v, cl[i].closestV, cl[i].dist_min)
                    self.updateCL_capacity(aux_solution, cl, cl[i].v)
                    i += 1
                    #c = cl.pop(vWithMaxCap)
                #of = (solution.instance.sortedDistances[0].distance-solution.instance.sortedDistances[-1].distance)/4
                stochastic_capacity = sum(solution.instance.capacity)

            
            
            capacity.append(stochastic_capacity)
            solution.stochastic_of["2"].append(of)
            
        #self.whitebox.fit_with_probability(self.whitebox.fit_logistic, prob='zero_at_t_max_decreasing_probability', max_time=self.simulations, time=_ + 1)    
        
        solution.reliability["2"] = (self.simulations - fail) / self.simulations
        solution.total_stochastic_capacity["2"] = np.mean(capacity)
        solution.mean_stochastic_of["2"] = np.mean(solution.stochastic_of["2"])
        

    def fast_simulation(self, solution: Solution):
        fail = 0
        dic_type = setCapType(solution)
       
        for _ in range(self.simulations):
            stochastic_capacity = 0
            for node in solution.selected:
                
                #capacity_node = self.random_value(solution.instance.capacity[node]) #solution.instance.capacity[node]
                cap_node = solution.instance.capacity[node]
                weather = solution.weather
                congestion = solution.congestion[node]
                probValue = self.blackbox.simulate(cap_node =cap_node, weather=weather, congestion=congestion, verbose=False)
                capacity_node =   probValue* solution.instance.capacity[node]#  
            #     #print("capacity_node:",capacity_node)
            #     # Record data into WhiteBox
                self.whitebox.add_data(cap_node =cap_node, weather=weather, congestion=congestion, variable=capacity_node)
                
                
                # if dic_type[node] == 0:
                #     capacity_node =  self.random_value(solution.instance.capacity[node])              
                    
                # elif dic_type[node] == 1:
                    
                #     cap_node = solution.instance.capacity[node]
                #     weather = solution.weather
                #     congestion = solution.congestion[node]
                #     probValue = self.blackbox.simulate(cap_node =cap_node, weather=weather, congestion=congestion, verbose=False)
                #     capacity_node =   probValue* solution.instance.capacity[node] 
                # #     #print("capacity_node:",capacity_node)
                # #     # Record data into WhiteBox
                #     self.whitebox.add_data(cap_node =cap_node, weather=weather, congestion=congestion, variable=capacity_node)
                    

                    
                    
                stochastic_capacity += capacity_node
            
            if stochastic_capacity < solution.instance.b:
                fail += 1
                solution.of = 0
                break  # Exit the loop immediately if capacity is insufficient
            
            #self.whitebox.fit_logistic_for_each_type()
        #self.whitebox.fit_with_probability(self.whitebox.fit_logistic, prob='zero_at_t_max_decreasing_probability', max_time=self.simulations, time=_ + 1)
            
        p = (self.simulations - fail) / self.simulations
        variance = ((p*(1-p))/self.simulations)**(1/2)
        inf = p-1.96*variance
        sup = p+1.96*self.var
        
        return (inf, sup)


    # def updateCL_capacity(self, sol, cl, lastAdded):
    #     instance = sol.instance
    #     for c in cl:
    #         dToLast = instance.distance[lastAdded][c.v]
            
    #         if dToLast < c.dist_min:
    #             c.dist_min = dToLast
    #             c.closestV = lastAdded

    #     self.max_min_dist = max([x.dist_min for x in cl])
    #     for c in cl:

    #         if self.max_min_dist != 0:
    #             c.cost = c.dist_min / self.max_min_dist * 0.8 + instance.capacity[c.v] / max(instance.capacity) * (1 - 0.8)
    #         else:
    #             c.cost = instance.capacity[c.v] / max(instance.capacity) * (1 - 0.8)
    #     cl.sort(key=lambda x: x.cost, reverse=True)  # ordena distancia de mayor a menos 


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
                c.cost = c.dist_min / self.max_min_dist * self.weight + instance.capacity[c.v] / self.max_capacity * (1 - self.weight)* self.whitebox.get_value_with_dict(cap_node, simulate)
            # If max_min_dist is zero, calculate the cost based on capacity only
                c.modified_capacity = instance.capacity[c.v] * self.whitebox.get_value_with_dict(cap_node, simulate)
            else:
                c.cost = instance.capacity[c.v] / self.max_capacity * (1 - self.weight)* self.whitebox.get_value_with_dict(cap_node, simulate)
                c.modified_capacity = instance.capacity[c.v] #* self.whitebox.get_value_with_dict(simulate)
                
        # Sort the candidate list based on the updated cost in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)
        




'''


#******dynamic solution
import math
import copy
import random
import numpy as np
from ConstructiveHeuristic import *
from scipy.stats import truncnorm
from objects import *

            
def setCapType(solution):
    # Dictionary to store the capacity type for each selected node in the solution
    dic_type = {}

    # Iterate through each selected node in the solution
    for node in solution.selected:
        # Check the remainder when dividing the node index by 3
        if node % 2 == 0:
            dic_type[node] = 1  # dynamic capacity type
        
        else:
            dic_type[node] = 0  # deterministic capacity type
    
    # Return the dictionary mapping node indices to their corresponding capacity types
    return dic_type              
  
        



class sim_learnheuristic:

    def __init__(self, simulations, variance):
        self.simulations = simulations
        self.var = variance
        self.blackbox = BlackBox()  # Initialize BlackBox instance
        self.whitebox = WhiteBox()  # Initialize WhiteBox instance 



    def simulation_1(self, solution: Solution):
        fail = 0
        capacity = []
        dic_type = setCapType(solution)
        solution.stochastic_of["1"] = solution.of # Optimizar
        
        for _ in range(self.simulations):
            stochastic_capacity = 0
            for node in solution.selected:
                
                weather = solution.weather
                congestion = solution.congestion[node]
                probValue = self.blackbox.simulate(weather=weather, congestion=congestion, verbose=False)
                capacity_node =   probValue* solution.instance.capacity[node]  #solution.instance.capacity[node]  
            #     #print("capacity_node:",capacity_node)
            #     # Record data into WhiteBox
                self.whitebox.add_data(weather=weather, congestion=congestion, variable=capacity_node)
                
                
                # if dic_type[node] == 0:
                #     seasonal_demand = random.uniform(-1, 1) # edge seasonal demand adversity level, between 0 (low) and 1 (high)
                #     operational_dis = random.uniform(-1, 1)
                    
                #      # For even IDs, consider them as dynamic
                #     capacity_node = getDynamicValue(node, solution, operational_dis, seasonal_demand)
                    
                # elif dic_type[node] == 1: # deterministic
                #      # For odd IDs, consider them as deterministic
                #     capacity_node = solution.instance.capacity[node]
                
                stochastic_capacity += capacity_node
                
            if stochastic_capacity < solution.instance.b:
                fail += 1
                

            capacity.append(stochastic_capacity)

        
        solution.mean_stochastic_of["1"] = ((self.simulations-fail)*solution.of)/self.simulations
        # Calculate reliability for scenario 1
        solution.reliability["1"] = (self.simulations - fail) / self.simulations
        # Calculate the mean of stochastic capacities and store it in total_stochastic_capacity
        solution.total_stochastic_capacity["1"] = np.mean(capacity)

    def simulation_2(self, solution: Solution, cl):
        fail = 0

        capacity = []
        solution.stochastic_of["2"] = []
        dic_type = setCapType(solution)
        #operational_dis = random.random()
        for _ in range(self.simulations):
            aux_solution = Solution(solution.instance)
            aux_solution.of = solution.of
            aux_solution.of = solution.vMin1
            aux_solution.of = solution.vMin2
            of = solution.of
            stochastic_capacity = 0
            for node in solution.selected:
                
                weather = solution.weather
                congestion = solution.congestion[node]
                probValue = self.blackbox.simulate(weather=weather, congestion=congestion, verbose=False)
                capacity_node =   probValue* solution.instance.capacity[node]  #solution.instance.capacity[node]  
            #     #print("capacity_node:",capacity_node)
            #     # Record data into WhiteBox
                self.whitebox.add_data(weather=weather, congestion=congestion, variable=capacity_node)
                
                # if dic_type[node] == 0:
                #     seasonal_demand = random.uniform(-1, 1) # edge seasonal demand adversity level, between 0 (low) and 1 (high)
                #     operational_dis = random.uniform(-1, 1)
                #     # For even IDs, consider them as dynamic
                #     capacity_node = getDynamicValue(node, solution, operational_dis, seasonal_demand)
                    
                # elif dic_type[node] == 1: # deterministic
                #     # For odd IDs, consider them as deterministic
                #     capacity_node = solution.instance.capacity[node]
                
                
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
                        #sol.updateOF(c.v, c.closestV, c.dist_min)
                        of = cl[i].dist_min
                    aux_solution.updateOF(cl[i].v, cl[i].closestV, cl[i].dist_min)
                    self.updateCL_capacity(aux_solution, cl, cl[i].v)
                    i += 1
                    #c = cl.pop(vWithMaxCap)
                #of = (solution.instance.sortedDistances[0].distance-solution.instance.sortedDistances[-1].distance)/4
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
                
                weather = solution.weather
                congestion = solution.congestion[node]
                probValue = self.blackbox.simulate(weather=weather, congestion=congestion, verbose=False)
                capacity_node =   probValue* solution.instance.capacity[node]  #solution.instance.capacity[node]  
            #     #print("capacity_node:",capacity_node)
            #     # Record data into WhiteBox
                self.whitebox.add_data(weather=weather, congestion=congestion, variable=capacity_node)
                
                # if dic_type[node] == 0:
                #     seasonal_demand = random.uniform(-1, 1) # edge seasonal demand adversity level, between 0 (low) and 1 (high)
                #     operational_dis = random.uniform(-1, 1)
                #     # For even IDs, consider them as dynamic
                #     capacity_node = getDynamicValue(node, solution, operational_dis, seasonal_demand)
                    
                # elif dic_type[node] == 1: # deterministic
                #     # For odd IDs, consider them as deterministic
                #     capacity_node = solution.instance.capacity[node]
                
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
            #open_type_node = open_type.copy()
            #open_type_node[self.dict_of_types[c.v]] += 1/self.instance.n
            simulate = { "weather": self.weather, "congestion": self.congestion[c.v]}
            # If the max_min_dist is not zero, calculate the cost with weight and capacity
            if self.max_min_dist != 0:
                c.cost = c.dist_min / self.max_min_dist * self.weight + instance.capacity[c.v] / self.max_capacity * (1 - self.weight)* self.whitebox.get_value_with_dict(simulate)
            # If max_min_dist is zero, calculate the cost based on capacity only
                c.modified_capacity = instance.capacity[c.v] * self.whitebox.get_value_with_dict(simulate)
            else:
                c.cost = instance.capacity[c.v] / self.max_capacity * (1 - self.weight)* self.whitebox.get_value_with_dict(simulate)
                c.modified_capacity = instance.capacity[c.v] #* self.whitebox.get_value_with_dict(simulate)
        # Sort the candidate list based on the updated cost in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)

    
'''
'''

#******stochastic solution
        


import copy
import random
import numpy as np
from ConstructiveHeuristic import *
import math
from scipy.stats import truncnorm


def setCapType(solution):
    # Dictionary to store the capacity type for each selected node in the solution
    dic_type = {}

    # Iterate through each selected node in the solution
    for node in solution.selected:
        # Check the remainder when dividing the node index by 3
        if node % 2 == 0:
            dic_type[node] = 1  # deterministic capacity type
        
        else:
            dic_type[node] = 0  # stochastic capacity type
    
    # Return the dictionary mapping node indices to their corresponding capacity types
    return dic_type  






class sim_learnheuristic:


    # Constructor with parameters simulations and variance
    def __init__(self, simulations, variance):
        # Initialize the simheuristic object with simulations and variance
        self.simulations = simulations
        self.var = variance

      # Method to generate a random value based on a given mean using lognormal distribution
    def random_value(self, mean):
        variance = self.var * mean
        mu = math.log(mean**2 / math.sqrt(variance + mean**2))
        sigma = math.sqrt(math.log(1 + variance / mean**2))
        return np.random.lognormal(mean=mu, sigma=sigma)
    
    #def random_value(self, mean):
    #    return np.random.lognormal(np.log(mean), self.var)

    
    # Method definition for simulation_1 in the simheuristic class
    def simulation_1(self, solution: Solution):
        # Initialize fail count and an empty list for capacity values
        fail = 0
        capacity = []
        dic_type = setCapType(solution)
        # Set the stochastic objective function for scenario 1 to the current objective function value
        solution.stochastic_of["1"] = solution.of # Optimizar
        #print(solution.selected)
        # Loop through the number of simulations
        for _ in range(self.simulations):
            
            # Initialize stochastic_capacity to zero for each simulation
            stochastic_capacity = 0
            #print(solution.selected)
            #print(solution.instance.b)
            #print([solution.instance.capacity[i] for i in solution.selected])
            #print(self.var)
            # Loop through selected nodes in the current solution
            for node in solution.selected:
                
                #capacity_node = self.random_value(solution.instance.capacity[node])
                #print(capacity_node, solution.instance.capacity[node])
                
                if dic_type[node] == 0:
                    capacity_node = solution.instance.capacity[node]
                    
                elif dic_type[node] == 1:
                    capacity_node = self.random_value(solution.instance.capacity[node])
                    
                    
                
                # Accumulate stochastic capacities
                stochastic_capacity += capacity_node 
            #print(solution.instance.capacity[node])
            # Check if the total stochastic capacity is less than the capacity constraint
            if stochastic_capacity < solution.instance.b:
                # Increment fail count if the constraint is not satisfied
                fail += 1
                
                
            # Append the stochastic_capacity value to the capacity list
            capacity.append(stochastic_capacity)
            
        solution.mean_stochastic_of["1"] = ((self.simulations-fail)*solution.of)/self.simulations
        # Calculate reliability for scenario 1
        solution.reliability["1"] = (self.simulations - fail) / self.simulations
        # Calculate the mean of stochastic capacities and store it in total_stochastic_capacity
        solution.total_stochastic_capacity["1"] = np.mean(capacity)
        







    # Method definition for simulation_2 in the simheuristic class
    def simulation_2(self, solution: Solution, cl):
        # Initialize fail count and an empty list for capacity values
        fail = 0
        capacity = []
        dic_type = setCapType(solution)
        # Initialize stochastic_of["2"] as a list in the solution object
        solution.stochastic_of["2"] = []
    
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
                #capacity_node = self.random_value(solution.instance.capacity[node])
                #print(capacity_node)
               
               if dic_type[node] == 0:
                    capacity_node = solution.instance.capacity[node]
                    
               elif dic_type[node] == 1:
                    capacity_node = self.random_value(solution.instance.capacity[node])
               
                     
                # Accumulate stochastic capacities
               stochastic_capacity += capacity_node 
    
            # Check if the total stochastic capacity is less than the capacity constraint
            if stochastic_capacity < solution.instance.b:
                # Increment fail count if the constraint is not satisfied
                fail += 1
                solution.of = 0
                #break  # Exit the loop immediately if capacity is insufficient
                # Initialize an index variable for iterating through candidate list (cl)
                i = 0
    
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





    def fast_simulation(self, solution: Solution):
        # Initialize the failure count
        fail = 0
        dic_type = setCapType(solution)
        # Perform stochastic simulations
        for _ in range(self.simulations):
            
            stochastic_capacity = 0
            
            # Simulate capacities for selected nodes
            for node in solution.selected:
                #capacity_node = self.random_value(solution.instance.capacity[node])
               
               
                if dic_type[node] == 0:
                    capacity_node = solution.instance.capacity[node]
                    
                elif dic_type[node] == 1:
                    capacity_node = self.random_value(solution.instance.capacity[node])
                    
                
                    
                stochastic_capacity += capacity_node
    
            # Check if the stochastic capacity is below the threshold
            if stochastic_capacity < solution.instance.b:
                fail += 1
                solution.of = 0
                #break  # Exit the loop immediately if capacity is insufficient
        # Calculate the success probability (p)
        p = (self.simulations - fail) / self.simulations
        
        # Calculate the variance
        variance = ((p * (1 - p)) / self.simulations) ** (1/2)
        
        # Calculate confidence intervals
        # The value 1.96 is commonly used for a 95% confidence interval in a normal distribution
        inf = p - 1.96 * variance
        sup = p + 1.96 * self.var  #self.var: desired level of confidence 
        
        # Return the lower and upper bounds of the success probability
        return (inf, sup)







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
            # If the max_min_dist is not zero, calculate the cost with weight and capacity
            if self.max_min_dist != 0:
                c.cost = c.dist_min / self.max_min_dist * 0.8 + instance.capacity[c.v] / max(instance.capacity) * (1 - 0.8)
            # If max_min_dist is zero, calculate the cost based on capacity only
            else:
                c.cost = instance.capacity[c.v] / max(instance.capacity) * (1 - 0.8)
        
        # Sort the candidate list based on the updated cost in descending order
        cl.sort(key=lambda x: x.cost, reverse=True)
  
'''