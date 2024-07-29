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



#class Probability:
#    ALWAYS_1 = "always_prob_1"
#    EXPONENTIAL = "decreasing_probability_exponencial"
#    LINEAR = "zero_at_t_max_decreasing_probability"


        
class BlackBox:
    def __init__(self, beta_0, beta_1, beta_2, beta_3):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.n = 1

    def __init__(self):
        self.n = 1
        self.beta_0 = 0.01 #0.005
        self.beta_1 = -0.7 # Random value between 0 and 4
        self.beta_2 = 0.8  # Random value between 0 and 4
        self.beta_3 = 1.0  # Random value between 0 and 4    

        
 



    def setter_betas(self, beta_0, beta_1, beta_2, beta_3):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.n = 1        


    # def setter_betas(self, beta_0=0, beta_1=1, beta_2=0.3, beta_3=0.5):
    #     self.beta_0 = beta_0
    #     self.beta_1 = beta_1
    #     self.beta_2 = beta_2
    #     self.beta_3 = beta_3
    #     self.n = len(beta_0)

    def get_value(self, cap_node =0, weather=0, congestion=0):
        #open_type_value = sum([i * j for i, j in zip(self.beta_1, open_type)])
        #open_type_value = sum([self.beta_1 * j for j in open_type])
        cap_node_value = self.beta_0 * cap_node
        weather_value = self.beta_2 * weather
        congestion_value = self.beta_3 * congestion
        exponent = self.beta_1 +  cap_node_value + weather_value + congestion_value
        if exponent < -100:
            exponent = -100

        return 1 / (1 + math.exp(-exponent))


    # def get_value_with_list(self, node_type, list_of_data):
    #     open_type_value = sum([i * j for i, j in zip(self.beta_1, list_of_data[0:self.n - 1])])
    #     weather_value = self.beta_2[node_type] * list_of_data[self.n - 1]
    #     congestion_value = self.beta_3[node_type] * list_of_data[self.n]
    #     exponent = self.beta_0[node_type] + weather_value + open_type_value + congestion_value

    #     return 1 / (1 + math.exp(-exponent))

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

    # def simulate_list(self, list_of_data):
    #     output = []
    #     for list in list_of_data:
    #         rand = random.random()
    #         node_type, data = list[0], list[1:]
    #         if rand > self.get_value_with_list(node_type, data):
    #             output.append(0)
    #         else:
    #             output.append(1)

    #     return output

    # def get_value_with_dict(self, dict_of_data):
    #     node_type = dict_of_data["node_type"]
    #     open_type_value = sum([i * j for i, j in zip(self.beta_1, dict_of_data["open_type"])])
    #     weather_value = self.beta_2[node_type] * dict_of_data["weather"]
    #     congestion_value = self.beta_3[node_type] * dict_of_data["congestion"]
    #     exponent = self.beta_0[node_type] + weather_value + open_type_value + congestion_value
    #     # Negativo = bueno
    #     if -exponent > 100:
    #         exponent = -100
    #     return 1 / (1 + math.exp(-exponent))

    # def simulate_dict(self, dict_of_data):
    #     output = []
    #     for list in dict_of_data:
    #         rand = random.random()
    #         if rand > self.get_value_with_dict(list):
    #             output.append(0)
    #         else:
    #             output.append(1)

    #     return output

    # def print_beta(self):
    #     for i in range(self.n):
    #         print("\n")
    #         print("Para el tipo de nodo " + str(i) + ":")
    #         print("Beta_0: " + str(self.beta_0[i]) + " correspondiente al término independiente")
    #         print("Beta_1: " + str(self.beta_1) + " correspondiente al término que acompaña al open_type")
    #         print("Beta_2: " + str(self.beta_2[i]) + " correspondiente al término que acompaña al weather")
    #         print("Beta_3: " + str(self.beta_3[i]) + " correspondiente al término que acompaña al congestion")
    #         print("\n")


class WhiteBox:
    def __init__(self):
        self.n = 1
        self.beta_0 = 0
        self.beta_1 = 0  # Random value between 0 and 4
        self.beta_2 = 0 # Random value between 0 and 4
        self.beta_3 = 0 # Random value between 0 and 4   
        self.data = []




    def get_value(self, cap_node=0, weather=0, congestion=0):
        #print("beta_0:", self.beta_0)
        #print("beta_1:", self.beta_1)
        #print("beta_2:", self.beta_2)
        #print("beta_3:", self.beta_3)
        #open_type_value = sum([self.beta_1 * j for j in open_type])
        cap_node_value = self.beta_0 * cap_node
        weather_value = self.beta_2 * weather
        congestion_value = self.beta_3 * congestion
        exponent = self.beta_1 +  cap_node_value + weather_value + congestion_value

        return 1 / (1 + math.exp(-exponent))




    def get_value_with_dict(self, cap_node, dict_values):

        #node_type = dict_values["node_type"]
        #open_type = dict_values["open_type"]
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

    # def simulate_dict(self, dict_values, verbose=False):
    #     rand = random.random()

    #     node_type = dict_values["node_type"]
    #     open_type = dict_values["open_type"]
    #     weather = dict_values["weather"]
    #     congestion = dict_values["congestion"]

    #     if verbose:
    #         print("La probabilidad de la white box ha dado: " + str(
    #             self.get_value(node_type, open_type, weather, congestion)))
    #         print("Y el número aleatorio ha dado: " + str(rand))

    #     if rand > self.get_value(node_type, open_type, weather, congestion):
    #         if verbose:
    #             print("Se pierde la capacidad del nodo")
    #         return 0
    #     else:
    #         if verbose:
    #             print("No se pierde la capacidad del nodo")
    #         return 1

    # def simulate_dict_list(self, dict_list, verbose=False):
    #     output = []
    #     for single_dict in dict_list:
    #         output.append(self.simulate_dict(single_dict, verbose))

    #     return output

    def add_data(self, cap_node, weather, congestion, variable):
        dato = [cap_node] + [weather] + [congestion] + [variable]
        self.data.append(dato)
        
    # def add_data_in_list(self, list_of_data: list):
    #     """
    #     :param list_of_data:
    #                         1º value: node_type
    #                         2º value: array of open_type N^n
    #                         3º value: weather value 1 or 0
    #                         4º value: congestion value 1 or 0
    #                         5º value: variable value 1 or 0
    #     :return:
    #     """
    #     if isinstance(list_of_data, Iterable) and not isinstance(list_of_data, str):
    #         if all(isinstance(sub_obj, Iterable) for sub_obj in list_of_data):
    #             for lista in list_of_data:
    #                 node_type = lista[0]
    #                 self.data[node_type].append(lista[1:])
    #         else:
    #             node_type = list_of_data[0]
    #             self.data[node_type].append(list_of_data[1:])
    #     else:
    #         print("WTF, me has pasado '" + list_of_data + "'. Espabila niño, que te la vida te va a comer")

    # def add_data_in_dict(self, dict_of_data: list):
    #     for data in dict_of_data:
    #         node_type = data["node_type"]
    #         dato = data["open_type"] + [data["weather"]] + [data["congestion"]] + [data["variable"]]
    #         self.data[node_type].append(dato)

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



    # def fit_logistic_for_each_type(self):
    #         if len(self.data) >= 2:
    #             x, y = zip(*([[data[:-1], data[-1]] for data in self.data]))
    #             if len(set(y)) != 1:
    #                 log_reg = LogisticRegression(max_iter=1000)
    #                 log_reg.fit(x, y)
    #                 coef = log_reg.coef_[0]
    #                 intercept = log_reg.intercept_[0]
    #                 self.beta_1 = coef[0:len(coef) - 1]
    #                 self.beta_1, self.beta_2, self.beta_3 = intercept, coef[-1], coef[-1]



    def fit_logistic(self):
        if len(self.data) >= 2:
            x, y = zip(*([[data[:-1], data[-1]] for data in self.data]))
            if len(set(y)) != 1:
                log_reg = LogisticRegression(max_iter=1000)
                log_reg.fit(x, y)
                coef = log_reg.coef_[0]
                intercept = log_reg.intercept_[0]
                
                self.beta_1, self.beta_0, self.beta_2, self.beta_3 = intercept, coef[-1], coef[-1], coef[-1]



    # def fit_logistic(self):
    #     result = [[[data[:-1], data[-1]] for data in self.data[j]] for j in self.data if self.data[j] != []]
    #     x, y = zip(*[(sublist[0], sublist[1]) for inner_list in result for sublist in inner_list])
    #     log_reg = LogisticRegression()
    #     if len(set(y)) != 1:
    #         log_reg.fit(x, y)
    #         intercept = log_reg.intercept_[0]
    #         coef = log_reg.coef_[0]
    #         self.beta_0 = {i: intercept for i in range(self.n)}
    #         self.beta_1 = {i: coef[0:self.n] for i in range(self.n)}
    #         self.beta_2 = {i: coef[self.n] for i in range(self.n)}
    #         self.beta_3 = {i: coef[-1] for i in range(self.n)}

    # def print_beta(self):
    #     for i in range(self.n):
    #         print("\n")
    #         print("Para el tipo de nodo " + str(i) + ":")
    #         print("Beta_1: " + str(self.beta_1[i]) + " correspondiente al término independiente")
    #         print("Beta_2: " + str(self.beta_2[i]) + " correspondiente al término que acompaña al weather")
    #         print("Beta_3: " + str(self.beta_3[i]) + " correspondiente al término que acompaña al congestion")
    #         print("\n")
 
    
 



    
    


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
    #beta0 = 0.01
    #beta1 = -1.5
    #beta2 = 1.5
    #beta3 = 1
    cap_node = 100
    blackbox = BlackBox()
    blackbox.setter_betas(beta0, beta1, beta2, beta3)
    plot_heat_map(blackbox, "1.png", cap_node)






