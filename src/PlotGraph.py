from Graph import Graph
from planar_graph import PlanarTriangulation

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing as mp
import math


def plot_graph(
    graph: Graph, name, labels_dict: dict | None = None
):

    g = nx.from_numpy_array(graph.to_adjacency_matrix())


    plt.figure(figsize=(8, 8))
    position= nx.planar_layout(g, scale = 1, center=(0, 0))
  
    filepath=""
    if labels_dict is not None:
        print(labels_dict)
        nx.draw(g, pos=position, labels = labels_dict, with_labels = True)
        filepath = 'graph_plots/cubic/'
    else:
        nx.draw(g, pos=position)
        nx.draw_networkx_labels(g, position)
        filepath = 'graph_plots/trian/'

    
    plt.savefig(filepath + name)

        
def plot_distance(hist: np.histogram, title: str, filepath: str | None = None):

    plt.plot(hist)
    
    plt.title('Histogram of number of immediate neighbours')
    
    plt.xlabel('Number of Occurrences', fontsize=10)
    plt.ylabel('Number of immediate neighbours', fontsize=10)
    
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()
    plt.clf()
    
def plot_distance_from_saved(name, pltname, N, col = 'blue', same_plot = False):
    
    with open('data/cubic/'+ name +'.npy', 'rb') as f:
        a = np.load(f)
    x = np.arange(0,len(a))
    x= x.astype(float)

    a = a/np.sum(a)
    x = (x+5.5)/(pow(N,  0.25)  - 0.45)
    plt.scatter(x, a, color=col)
    
    plt.xlim(0, 20)
    plt.title('Histogram odległości')
    
    plt.xlabel('Odległość', fontsize=10)
    plt.ylabel('Udział', fontsize=10)
    plt.savefig('histograms/cubic/' + pltname)
    
    if not same_plot:
        plt.clf()
        plt.close()
        
        
def plot_neighbours_from_saved(name, pltname, log=False ,same_plot = False):
    
    with open('data/trian/'+ name +'.npy', 'rb') as f:
        a = np.load(f)
        
    unique_values, counts = np.unique(a, return_counts=True)
    counts = np.divide(counts, np.sum(counts))
    
    theory = np.ndarray( np.max(unique_values) + 1, np.float64)
    theory[0] = None
    theory[1] = None
    theory[2] = None
    
    for k in range(3, theory.shape[0]):
        theory[k] = 16*(pow((3.0/16), k))*(k-2)*math.factorial((2*k-2))/(math.factorial(k)*math.factorial((k-1)))
    

    theory_x = np.arange(0, theory.shape[0])

    plt.scatter(unique_values, counts, color='blue')
    plt.plot(theory_x, theory, color='magenta')
    
    plt.title('Histogram sąsiadów')
    
    plt.ylabel('Udział', fontsize=10)
    plt.xlabel('Liczba sąsiadów', fontsize=10)
    plt.savefig('histograms/trian/' + pltname)
    plt.clf()
    
    counts =-np.log10(counts)
    theory =-np.log10(theory)
    
    plt.scatter(unique_values, counts, color='blue')
    plt.plot(theory_x, theory, color='magenta')
    
    plt.title('Histogram sąsiadów')
    
    plt.ylabel('log10(Udział)', fontsize=10)
    plt.xlabel('Liczba sąsiadów', fontsize=10)
    plt.savefig('histograms/trian/log' + pltname)
    
    plt.clf()
    plt.close()
    

        
   
    
def plot_multiple_neigh(name: str, *data_sets):
    plt.clf()
    for i in range(len(data_sets)):
        plot_neighbours_from_saved(data_sets[i], name, True)
    plt.clf()
    plt.close()
    
def plot_multiple_dist(name: str, *data_sets):
    plt.clf()
    for i in range(len(data_sets)):
        plot_distance_from_saved(data_sets[i], name, True)
    plt.clf()
    plt.close()

   

    
def proc_acquire(args):
    num_of_nodes, num_of_sweeps, process_index, total_processes = args
    plg = PlanarTriangulation.random_planar_triangulation(num_of_nodes)
    hist = plg.number_of_neighbours()
    levels = np.zeros(num_of_nodes, np.int64)
    levels = plg.graph_BFS(levels, num_of_nodes)

    start = time.time()
    plg.sweep()
    end = time.time()
    
    for i in range(int(num_of_sweeps/total_processes)):
        # print("Get neighbours")
        hist = np.append(hist, plg.number_of_neighbours())
        # print("Get levels")
        levels = plg.graph_BFS(levels, num_of_nodes)
        print("Sweeping ", i+2)
        plg.sweep()

    g = nx.from_numpy_array(plg.to_adjacency_matrix())
    if not nx.is_planar(g):
        raise ValueError()
    return hist, levels

def acquire_data_proc(num_of_nodes, num_of_iters, num_of_processes, name: str | None = None):
    cores_num =  mp.cpu_count()
    if num_of_processes > cores_num:
        msg = "Sorry, it is not optimal to run more processes than half of how many CPU cores you have! You have " + cores_num + " CPU cores"
        raise Exception(msg)
    
    process_args = [(num_of_nodes, num_of_iters, i, num_of_processes) for i in range(num_of_processes)]

    with mp.Pool(processes=num_of_processes) as pool:
        results = pool.map(proc_acquire, process_args)

    hist = np.array([], np.int64)
    levels = np.zeros(num_of_nodes, np.int64)
    for i, (neigh, nlvl) in enumerate(results):
        hist = np.append(hist, neigh)
        levels = np.add(nlvl, levels)
    if not name:
        name = ""
    n_name = name + str(num_of_nodes) + 'x' + str(num_of_iters)
    d_name = name + str(num_of_nodes) + 'x' + str(num_of_iters)
    
    with open('data/trian/'+ n_name +'.npy', 'wb') as f:
        np.save(f, hist)
    with open('data/cubic/'+ d_name +'.npy', 'wb') as f:
        np.save(f, levels)
        
    plot_neighbours_from_saved(n_name, n_name)
    plot_distance_from_saved(n_name, n_name, num_of_nodes)



def acquire_data(num_of_nodes, num_of_iters, name) -> np.array:
    plg = PlanarTriangulation.random_planar_triangulation(num_of_nodes)

    hist = plg.number_of_neighbours()
    levels = np.zeros(num_of_nodes, np.int64)
    levels = plg.graph_BFS(levels, num_of_nodes)
    print("first levels", levels)
    print("First sweep")
    print("Sweeping ", 1)
    start = time.time()
    plg.sweep()
    end = time.time()
    print(end - start)
    
    for i in range(num_of_iters - 1):
        # print("Get neighbours")
        hist = np.append(hist, plg.number_of_neighbours())
        # print("Get levels")
        levels = plg.graph_BFS(levels, num_of_nodes)
        print("Sweeping ", i+2)
        plg.sweep()


    g = nx.from_numpy_array(plg.to_adjacency_matrix())
    if not nx.is_planar(g):
        raise ValueError()
    
    if not name:
        name = ""
        
    n_name = name + str(num_of_nodes) + 'x' + str(num_of_iters)
    d_name = name + str(num_of_nodes) + 'x' + str(num_of_iters)
    
    with open('data/trian/'+ n_name +'.npy', 'wb') as f:
        np.save(f, hist)
    with open('data/cubic/'+ d_name +'.npy', 'wb') as f:
        np.save(f, levels)
        
    plot_neighbours_from_saved(n_name, n_name)
    plot_distance_from_saved(n_name, n_name, num_of_nodes)



