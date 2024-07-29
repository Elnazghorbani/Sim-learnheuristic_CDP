import random
import math
from collections.abc import Iterable
import numpy as np
import inspect
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Candidate:

    def __init__(self, v, closestV, cost):
        self.v = v # ID
        self.closestV = closestV # node with the minmum distance with this candidate
        self.cost = cost # This cost just return the cost of each node considering the distances from other nodes
        self.type=  0
        
 
        
'''
this candidate is different from the previous candidate. In this type of candidate in addition to cost, the dist_min
is also considered as an attribute
'''

class Candidate_capacity:
    
    def __init__(self, v, closestV, dist_min, cost):
        self.v = v # ID
        self.closestV = closestV # node with the minmum distance with this candidate
        self.dist_min = dist_min # The distance between the closest node and this candidate
        self.cost = cost # The cost here contains distance and also capacity of the candidate
        



class Edge:
    def __init__(self, v1, v2, distance):
        # Initialize the Edge object with two vertices and a distance
        self.v1 = v1  # Vertex 1 of the edge
        self.v2 = v2  # Vertex 2 of the edge
        self.distance = distance  # Distance between the two vertices



#contain the parameters of the execution
class Test:
    def __init__(self, instName, seed, time, beta1, beta2, maxIter, delta, short_simulation, long_simulation, var, deterministic, not_penalization_cost, weight, inversa):
        self.instName = instName #Instance Name
        self.Maxtime = int(time) #max Execution Time
        self.betaBR = float(beta1) #beta BR 1
        self.betaLS = float(beta2) #beta BR 2
        self.seed = int(seed) #seed
        self.maxIter = int(maxIter) #seed
        self.delta = float(delta)
        self.short_simulation = int(short_simulation)
        self.long_simulation = int(long_simulation)
        self.var = float(var)
        self.deterministic = True if deterministic == "True" else False
        self.not_penalization_cost = True if not_penalization_cost == "True" else False
        self.weight = float(weight)
        self.inversa = float(inversa)


        
class BlackBox:
    def __init__(self, beta_0, beta_1, beta_2, beta_3):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.n = 1

    def __init__(self):
        self.n = 1
        self.beta_0 = 0.01 
        self.beta_1 = -0.7 
        self.beta_2 = 0.8  
        self.beta_3 = 1.0     

        
 



    def setter_betas(self, beta_0, beta_1, beta_2, beta_3):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.n = 1        




    def get_value(self, cap_node =0, weather=0, congestion=0):
        cap_node_value = self.beta_0 * cap_node
        weather_value = self.beta_2 * weather
        congestion_value = self.beta_3 * congestion
        exponent = self.beta_1 +  cap_node_value + weather_value + congestion_value
        if exponent < -100:
            exponent = -100

        return 1 / (1 + math.exp(-exponent))




    def simulate(self, cap_node= 0, weather=0, congestion=0, verbose=False):
        rand = random.random()

        if verbose:
            print("The black box probability has been: " + self.get_value(cap_node, weather, congestion))

        if rand > self.get_value(cap_node, weather, congestion):
            if verbose:
                print("Node capacity is lost")
            return 0
        else:
            if verbose:
                print("Node capacity is not lost")
            return 1


class WhiteBox:
    def __init__(self):
        self.n = 1
        self.beta_0 = 0
        self.beta_1 = 0  
        self.beta_2 = 0 
        self.beta_3 = 0  
        self.data = []




    def get_value(self, cap_node=0, weather=0, congestion=0):
        #open_type_value = sum([self.beta_1 * j for j in open_type])
        cap_node_value = self.beta_0 * cap_node
        weather_value = self.beta_2 * weather
        congestion_value = self.beta_3 * congestion
        exponent = self.beta_1 +  cap_node_value + weather_value + congestion_value

        return 1 / (1 + math.exp(-exponent))




    def get_value_with_dict(self, cap_node, dict_values):

        weather = dict_values["weather"]
        congestion = dict_values["congestion"]
     
        #open_type_value = sum([self.beta_1 * j for j in open_type])
        cap_node_value = self.beta_0 * cap_node
        weather_value = self.beta_2 * weather
        congestion_value = self.beta_3 * congestion
        exponent = self.beta_1 + cap_node_value + weather_value + congestion_value

        return 1 / (1 + math.exp(-exponent))

    def simulate(self, cap_node=0, weather=0, congestion=0, verbose=False):
        rand = random.random()
        if verbose:
            print("La probabilidad de la white box ha dado: " + str(
                self.get_value(cap_node, weather, congestion)))
            print("Y el número aleatorio ha dado: " + str(rand))

        if rand > self.get_value(cap_node, weather, congestion):
            if verbose:
                print("Se pierde la capacidad del nodo")
            return 0
        else:
            if verbose:
                print("No se pierde la capacidad del nodo")
            return 1


    def add_data(self, cap_node, weather, congestion, variable):
        dato = [cap_node] + [weather] + [congestion] + [variable]
        self.data.append(dato)
        
    
    def fit(self, fit_function):
        fit_function()

    def zero_at_t_max_decreasing_probability(self, max_time, time):
        return 1 - (time / max_time) if time < max_time else 0

    def decreasing_probability_exponencial(self, max_time, time):
        return np.exp(np.log(0.01) / max_time * time)

    def always_prob_1(self):
        return 1

    def fit_with_probability(self, fit_function, prob, **kwargs):
        try:
            func_name = getattr(Probability, prob)
            method_to_call = getattr(self, func_name, None)
        except AttributeError:
            func_name = "always_prob_1"
            method_to_call = getattr(self, func_name, None)

        params = inspect.signature(method_to_call).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        prob = method_to_call(**filtered_kwargs)
        random_number = random.random()
        if random_number < prob:
            fit_function()



    def fit_logistic(self):
        if len(self.data) >= 2:
            x, y = zip(*([[data[:-1], data[-1]] for data in self.data]))
            if len(set(y)) != 1:
                log_reg = LogisticRegression(max_iter=1000)
                log_reg.fit(x, y)
                coef = log_reg.coef_[0]
                intercept = log_reg.intercept_[0]
                
                self.beta_1, self.beta_0, self.beta_2, self.beta_3 = intercept, coef[-1], coef[-1], coef[-1]

  
def plot_heat_map(bb: BlackBox, output_name, cap_node):
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 8))
    weather_values = [0, 1]
    congestion_values = [0, 1]
    data = []
    for w in weather_values:
        row = []
        for c in congestion_values:
            row.append(bb.get_value(cap_node, w, c))
        data.append(row)
    
    sns.heatmap(data, annot=True, ax=ax, cbar=True, xticklabels=congestion_values, yticklabels=weather_values,
                cmap="coolwarm", annot_kws={"size": 28})
    ax.set_xlabel('Seasonal demand', fontsize=28)
    ax.set_ylabel('Operational disruption', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)  # Set font size for both axes

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)  # Set font size for colorbar

    plt.tight_layout()

    # Create output folder if it doesn't exist
    output_folder = '../../output/figures'
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, output_name)
    
    # Save the heatmap
    plt.savefig('h', dpi=500)
    
    # Optionally, show the plot
    plt.show()
    
    plt.close(fig)

def plot_wit_some_betas():
    beta0 = 0.01
    beta1 = -0.7
    beta2 = 0.8
    beta3 = 1.0
    cap_node = 100
    blackbox = BlackBox()
    blackbox.setter_betas(beta0, beta1, beta2, beta3)
    plot_heat_map(blackbox, "1.png", cap_node)

