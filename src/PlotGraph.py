from Graph import Graph
from planar_graph import PlanarTriangulation

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time



def plot_graph(
    graph: Graph, filepath: str | None = None, labels_dict: dict | None = None, sp: Graph | None = None
):
    #print(graph.to_adjacency_matrix())
    g = nx.from_numpy_array(graph.to_adjacency_matrix())
    if sp is not None:
        s = nx.from_numpy_array(sp.to_adjacency_matrix())
    #pos = nx.spring_layout(g, center=(0, 0))

    plt.figure(figsize=(15, 15))
    position= nx.planar_layout(g, scale = 1, center=(0, 0))
  
    if labels_dict is not None:
        #nx.draw_networkx_labels(labels_dict, position)
        nx.draw(g, pos=position, labels = labels_dict, with_labels = True)
    else:
        nx.draw(g, pos=position)
        nx.draw_networkx_labels(g, position)

    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()
        
def plot_histogram(hist: np.histogram, title: str, filepath: str | None = None):
    h, bin_edges = np.histogram(hist)
    plt.figure(figsize=[10,8])
    plt.hist(hist, bins=bin_edges)
    # plt.bar(bin_edges[:-1], hist, width = 0.5, color='#0504aa',alpha=0.7)
    # plt.xlim(min(bin_edges), max(bin_edges))
    # plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Normal Distribution Histogram ' + title, fontsize=15)
    
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()


def acquire_data(num_of_nodes, num_of_sweeps, filepath: str | None = None):
    print("Generating random planar triangulation")
    plg = PlanarTriangulation.random_planar_triangulation(num_of_nodes)
    
    hist = plg.number_of_neighbours()
    levels = plg.graph_BFS()
    print("First sweep")
    print("Sweeping ", 1)
    plg.sweep()
    for i in range(num_of_sweeps-1):
        print("Get neighbours")
        hist = np.append(hist, plg.number_of_neighbours())
        print("Get levels")
        levels =  np.append(levels, plg.graph_BFS())
        print("Sweeping ", i+2)
        plg.sweep()

    if filepath:
        pth = filepath
    else:
        pth= ""
        
    levels = np.delete(levels, np.where(levels == 0))
    with open('data/trian/'+ pth + str(num_of_nodes) + 'x' + str(num_of_sweeps) +'.npy', 'wb') as f:
        np.save(f, hist)
        
    with open('data/cubic/'+ pth + str(num_of_nodes) + 'x' + str(num_of_sweeps) +'.npy', 'wb') as f:
        np.save(f, hist)
    
    
    plot_histogram(hist, "Neighbours histogram","histograms/trian/" + pth + str(num_of_nodes) + 'x' + str(num_of_sweeps))
    plot_histogram(levels, "Levels histogram", "histograms/cubic/" + pth + str(num_of_nodes) + 'x' + str(num_of_sweeps))

acquire_data(50, 100)







