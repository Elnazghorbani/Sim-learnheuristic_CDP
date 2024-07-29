import random
class Solution:

   

    def __init__(self, instance):
        # Initialize the solution with the given instance
        self.instance = instance
        self.selected = [] # List to store the selected nodes in the solution
        # Initialize vMin1 and vMin2 to -1
        self.vMin1 = -1
        self.vMin2 = -1
        # Set the objective function value (of) to 10 times the distance of the first sorted distance
        self.of = instance.sortedDistances[0].distance * 10
        self.capacity = 0
        self.time = 0
        # Dictionary to store reliability values for scenarios 1 and 2
        self.reliability = {"1": 0, "2": 0}
        # Dictionary to store the total stochastic capacity for scenarios 1 and 2
        self.total_stochastic_capacity = {"1": 0, "2": 0}
        # Dictionary to store stochastic objective function values for scenarios 1 and 2
        self.stochastic_of = {"1": 0, "2": 0}
        # Dictionary to store mean stochastic objective function values for scenarios 1 and 2
        self.mean_stochastic_of = {"1": 0, "2": 0}
        self.weather = random.randint(0, 1)
        self.congestion = {i: random.randint(0, 1) for i in range(instance.n)}


    def copySol(self):
        # Create a new solution instance with the same problem instance as the current solution
        newSol = Solution(self.instance)
        # Copy values of vMin1, vMin2, objective function (of), capacity, and time from the current solution
        newSol.vMin1 = self.vMin1
        newSol.vMin2 = self.vMin2
        newSol.of = self.of
        newSol.capacity = self.capacity
        newSol.time = self.time

        # Copy reliability, total stochastic capacity, stochastic objective function, and mean stochastic objective function for scenarios 1 and 2
        for i in range(2):
            newSol.reliability[str(i + 1)] = self.reliability[str(i + 1)]
            newSol.total_stochastic_capacity[str(i + 1)] = self.total_stochastic_capacity[str(i + 1)]
            newSol.stochastic_of[str(i + 1)] = self.stochastic_of[str(i + 1)]
            newSol.mean_stochastic_of[str(i + 1)] = self.mean_stochastic_of[str(i + 1)]

        # Copy the list of selected nodes from the current solution to the new solution
        for i in self.selected:
            newSol.selected.append(i)

        # Return the new solution with copied attributes
        return newSol



    def add(self, v):
        # Add the node 'v' to the list of selected nodes
        self.selected.append(v)
        # Update the total capacity by adding the capacity of the newly selected node
        self.capacity += self.instance.capacity[v]


    def drop(self, v):
        # Find the index of the node 'v' in the list of selected nodes
        index = self.selected.index(v)
        # Remove the node 'v' from the list of selected nodes
        del self.selected[index]
        # Update the total capacity by subtracting the capacity of the dropped node
        self.capacity -= self.instance.capacity[v]




    # Function to calculate the distance from a vertex to the selected vertices in the solution
    def distanceTo(self, v):
        # Initialize the minimum distance to a large value
        minDist = self.instance.sortedDistances[0].distance * 10
        
        # Initialize the closest vertex to -1
        vMin = -1
        
        # Iterate through the selected vertices in the solution
        for s in self.selected:
            # Calculate the distance from the selected vertex (s) to the target vertex (v)
            d = self.instance.distance[s][v]
            
            # If the calculated distance is smaller than the current minimum distance
            if d < minDist:
                # Update the minimum distance and the closest vertex
                minDist = d
                vMin = s
        
        # Return the closest vertex and the minimum distance to that vertex
        return vMin, minDist
    


    #This function defines the feasibility of a solution
    def isFeasible(self):
        #if the capacity of a solution is bigger than the demand, the solution is feasible
        return self.capacity >= self.instance.b

    #This function gives us the updated objective function
    def updateOF(self,vMin1,vMin2, of):
        self.of = of
        self.vMin1 = vMin1
        self.vMin2 = vMin2



    def getEvalComplete(self):
        # Initialize the objective function value with a large initial value
        self.of = self.instance.sortedDistances[0].distance * 10

        # Iterate through all pairs of selected nodes
        for s1 in self.selected:
            for s2 in self.selected:
                # Skip pairs with the same node
                if s1 == s2:
                    continue

                # Update the objective function value if the distance between the pair is smaller
                d = self.instance.distance[s1][s2]
                if d < self.of:
                    self.of = d

        # Return the updated objective function value
        return self.of




    #The objective function value represents the minimum distance between any two selected vertices in the solution. 
    def reevaluateSol(self):
        # Initialize the objective function value with a large initial value
        self.of = self.instance.sortedDistances[0].distance * 10
    
        # Iterate through pairs of selected vertices in the solution
        for s1 in self.selected:
            for s2 in self.selected:
                # Skip pairs where both vertices are the same
                if s1 == s2:
                    continue
    
                # Calculate the distance between the pair of vertices
                d = self.instance.distance[s1][s2]
    
                # Update the objective function value if the calculated distance is smaller
                if d < self.of:
                    self.of = d
                    self.vMin1 = s1
                    self.vMin2 = s2
