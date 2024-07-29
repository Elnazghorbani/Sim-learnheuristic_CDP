import numpy as np
from objects import Edge



class Instance:
    def __init__(self, path):
        # Initialize instance attributes
        self.name = ""            # Placeholder for instance name
        self.n = 0                # Placeholder for the number of nodes
        self.b = 0                # Placeholder for the minimum required capacity
        self.capacity = []        # List to store the capacity of each node
        self.distance = None       # Placeholder for distance matrix
        self.sortedDistances = []  # List to store sorted distances
        # Call the readInstance method to populate instance attributes
        self.readInstance(path)

        
        
        
    def readInstance(self, s):
        # Open the file specified by the path 's' for reading
        with open(s) as instance:
            i = 1           # Counter for lines
            fila = 0        # Counter for rows in the distance matrix
            # Iterate through each line in the file
            for line in instance:
                if line == "\n":
                    continue  # Skip empty lines
                if i == 1:    # First line: number of nodes
                    self.n = int(line)
                    self.distance = np.zeros((self.n, self.n))  # Initialize distance matrix with zeros
                elif i == 2:  # Second line: required capacity
                    self.b = int(line)
                elif i == 3:  # Third line: node capacities
                    l = line.rstrip('\t\n ')
                    self.capacity = [float(x) for x in l.split('\t')]  # Convert capacities to a list of floats
                else:          # Lines after the third: distance matrix
                    l = line.rstrip('\t\n ')
                    d = [float(x) for x in l.split('\t')]  # Convert distances to a list of floats
                    for z in range(0, self.n):
                        if d[z] != 0:
                            # Populate the distance matrix and sortedDistances list with Edge objects
                            self.distance[fila, z] = d[z]
                            self.sortedDistances.append(Edge(fila, z, d[z]))
                    fila += 1  # Increment the row counter
                i += 1  # Increment the line counter
        # Sort the sortedDistances list based on distance in descending order
        self.sortedDistances.sort(key=lambda x: x.distance, reverse=True)
