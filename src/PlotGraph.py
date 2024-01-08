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
    position2= nx.planar_layout(g, scale = 1, center=(0, 0.03))
  
    filepath=""
    filepath=""
    if labels_dict is not None:
        print(labels_dict)
        # nx.draw(g, pos=position2, labels = labels_dict, with_labels = True)
        nx.draw(g, pos=position)
        nx.draw_networkx_labels(g, position2, labels = labels_dict, font_color='red')
        filepath = 'graph_plots/cubic/'
    else:
        nx.draw(g, pos=position)
        nx.draw_networkx_labels(g, position)
        filepath = 'graph_plots/trian/'

    
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
    
    with open('data/distance/'+ name +'.npy', 'rb') as f:
        a = np.load(f)
    x = np.arange(0,len(a))
    x= x.astype(float)
    a= a.astype(float)

    a = a/np.sum(a)
    # x = (x+2)/(pow(N,  0.25)-1.15- 0.45)
    # a = (a)*( pow(N,  0.25) -1.15- 0.45)
    # v = 1
    # x = (x + 0.1)/(pow(N,  0.25) - v)
    # a = (a)*( pow(N,  0.25) - v)
    print(a.tolist())
    plt.scatter(x, a, color=col)
    
    plt.xlim(0, 25)
    plt.title('Histogram odległości')
    
    # plt.xticks(np.arange(0, 21 , 1))
    plt.xlabel('r', fontsize=10)
    plt.ylabel('h(r)', fontsize=10)
    # plt.xlabel('x', fontsize=10)
    # plt.ylabel('f(x)', fontsize=10)
    plt.savefig('histograms/distance/' + pltname)
    
    if not same_plot:
        plt.clf()
        plt.close()
        
        
def plot_neighbours_from_saved_normal(name, pltname, col='blue', same_plot = False):
    
    with open('data/neighbours/'+ name +'.npy', 'rb') as f:
        a = np.load(f)
        
    unique_values, counts = np.unique(a, return_counts=True)
    counts = np.divide(counts, np.sum(counts))
    
    # theory = np.ndarray( np.max(unique_values) + 1, np.float128)
    theory = np.ndarray( 41, np.float128)
    theory[0] = None
    theory[1] = None
    theory[2] = None
    
    # for k in range(3, theory.shape[0]):
    for k in range(3, 41):
        theory[k] = 16*((3.0/16)**k)*(k-2)*math.factorial((2*k-2))/(math.factorial(k)*math.factorial((k-1)))
    

    # theory_x = np.arange(0, theory.shape[0])
    theory_x = np.arange(0, 41)


    plt.scatter(unique_values, counts, color=col)
    plt.plot(theory_x, theory, color='magenta')
    
    plt.xlim(0, 41)
    plt.title('Histogram sąsiadów')
    plt.ylabel('Udział', fontsize=10)
    plt.xlabel('Liczba sąsiadów', fontsize=10)
    plt.savefig('histograms/neighbours/normal/' + pltname)
    
    
    if not same_plot:
        plt.clf()
        plt.close()
    

def plot_neighbours_from_saved_log(name, pltname, col='blue', same_plot = False):
    with open('data/neighbours/'+ name +'.npy', 'rb') as f:
        a = np.load(f)
        
    unique_values, counts = np.unique(a, return_counts=True)
    counts = np.divide(counts, np.sum(counts))
    
    # theory = np.ndarray( np.max(unique_values) + 1, np.float64)
    theory = np.ndarray( 44, np.float64)
    theory[0] = None
    theory[1] = None
    theory[2] = None
    
    # for k in range(3, theory.shape[0]):
    for k in range(3, 44):
        theory[k] = 16*((3.0/16)**k)*(k-2)*math.factorial((2*k-2))/(math.factorial(k)*math.factorial((k-1)))
    

    # theory_x = np.arange(0, theory.shape[0])
    theory_x = np.arange(0, 44)
    
    counts =-np.log10(counts)
    theory =-np.log10(theory)
    
    plt.xlim(0, 44)
    plt.scatter(unique_values, counts, color=col)
    plt.plot(theory_x, theory, color='magenta')
    
    plt.title('Histogram sąsiadów')

    plt.ylabel('-log10(Udział)', fontsize=10)
    plt.xlabel('Liczba sąsiadów', fontsize=10)
    plt.savefig('histograms/neighbours/log/' + pltname)
    
    if not same_plot:
        plt.clf()
        plt.close()

   

    
def proc_acquire(args):
    num_of_nodes, num_of_sweeps, process_index, total_processes = args
    plg = PlanarTriangulation.random_planar_triangulation(num_of_nodes)
    
    hist = plg.number_of_neighbours()
    
    g, d= plg.find_cubic_dual_to_triangulation()
    # num_of_nodes_c = g._adjacency_matrix.shape[0]
    levels = np.zeros(num_of_nodes, np.int64)
    levels = plg.graph_BFS(levels, num_of_nodes)


    
    for i in range(100):
        print("Initial Sweep ", i)
        plg.sweep()
    
    for i in range(int(num_of_sweeps/total_processes) - 1):
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
    
    plg = PlanarTriangulation.random_planar_triangulation(num_of_nodes)
    g, d = plg.find_cubic_dual_to_triangulation()
    # num_of_nodes_c = g._adjacency_matrix.shape[0]
    
    process_args = [(num_of_nodes, num_of_iters, i, num_of_processes) for i in range(num_of_processes)]

    with mp.Pool(processes=num_of_processes) as pool:
        results = pool.map(proc_acquire, process_args)

    hist = np.array([], np.int64)
    levels = np.zeros(num_of_nodes, np.int64)
    for i, (neigh, nlvl) in enumerate(results):
        hist = np.append(hist, neigh)
        levels = np.add(nlvl, levels)
    # for i, nlvl in enumerate(results):
    #     # hist = np.append(hist, neigh)
    #     levels = np.add(nlvl, levels)
    if not name:
        name = ""
    n_name = name + str(num_of_nodes) + 'x' + str(num_of_iters)
    d_name = name + str(num_of_nodes) + 'x' + str(num_of_iters)
    
    with open('data/neighbours/'+ n_name +'.npy', 'wb') as f:
        np.save(f, hist)
    with open('data/distance/'+ d_name +'.npy', 'wb') as f:
        np.save(f, levels)
        
    plot_neighbours_from_saved_normal(n_name, n_name)
    plot_neighbours_from_saved_log(n_name, n_name)
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
    
    with open('data/neighbours/'+ n_name +'.npy', 'wb') as f:
        np.save(f, hist)
    with open('data/distance/'+ d_name +'.npy', 'wb') as f:
        np.save(f, levels)
        
    plot_neighbours_from_saved_normal(n_name, n_name)
    plot_neighbours_from_saved_log(n_name, n_name)
    plot_distance_from_saved(n_name, n_name, num_of_nodes)



# acquire_data_proc(33, 100, 5, 'cubasdic')

# plot_distance_from_saved("cubic64x32000", "n_merge_cubic", 124, col='blue', same_plot=True)
# plot_distance_from_saved("cubic128x16000", "n_merge_cubic", 252, col='orange', same_plot=True)
# plot_distance_from_saved("cubic256x8000", "n_merge_cubic", 508, col='yellow', same_plot=True)
# # plot_distance_from_saved("cubic512x4000", "merge_cubic", 1020, col='green', same_plot=True)
# plot_distance_from_saved("cubic1024x2000", "n_merge_cubic", 2044, col='pink', same_plot=True)
# plot_distance_from_saved("cubic2048x1000", "n_merge_cubic", 4092, col='black', same_plot=True)
# plt.legend(["124", "252", "508", "508", "2044", "4092"], loc ="lower right") 
# plt.savefig('histograms/distance/n_merge_cubic')
# plt.clf()
# plt.close()
# plot_distance_from_saved("cubic4096x500", "merge", 4096, col='magenta', same_plot=False)
# x = np.arange(0, 6, 0.1, dtype=float)
# alpha = 2.8
# y = 2./(alpha*alpha)*x*x*x*(np.exp(-x*x/(2*alpha)))

plot_distance_from_saved("method64x32000", "n_merge", 64, col='blue', same_plot=True)
plot_distance_from_saved("method128x16000", "n_merge", 128, col='orange', same_plot=True)
plot_distance_from_saved("method256x8000", "n_merge", 256, col='yellow', same_plot=True)
plot_distance_from_saved("method512x4000", "n_merge", 512, col='green', same_plot=True)
plot_distance_from_saved("method1024x2000", "n_merge", 1024, col='pink', same_plot=True)
plot_distance_from_saved("method2048x1000", "n_merge", 2048, col='black', same_plot=True)
plt.legend(["64", "128", "256", "512", "1024", "2048"], loc ="lower right") 
# plt.plot(x, y, color = 'magenta')
plt.savefig('histograms/distance/n_merge')
plt.clf()
plt.close()
# plot_distance_from_saved("cubic4096x500", "merge", 4096, col='magenta', same_plot=False)

# plot_neighbours_from_saved_normal("method64x32000", "merge", col='blue', same_plot=True)
# plot_neighbours_from_saved_normal("method128x16000", "merge", col='orange', same_plot=True)
# plot_neighbours_from_saved_normal("method256x8000", "merge", col='yellow', same_plot=True)
# plot_neighbours_from_saved_normal("method512x4000", "merge", col='green', same_plot=True)
# plot_neighbours_from_saved_normal("method1024x2000", "merge", col='pink', same_plot=True)
# plot_neighbours_from_saved_normal("method2048x1000", "merge", col='black', same_plot=True)
# plot_neighbours_from_saved_normal("method4096x500", "merge", col='magenta', same_plot=False)

# plot_neighbours_from_saved_log("method64x32000", "method64x32000", col='blue', same_plot=False)
# plot_neighbours_from_saved_log("method128x16000", "method128x16000", col='blue', same_plot=False)
# plot_neighbours_from_saved_log("method256x8000", "method256x8000", col='blue', same_plot=False)
# plot_neighbours_from_saved_normal("method256x8000", "method256x8000", col='blue', same_plot=False)
# plot_neighbours_from_saved_log("method512x4000", "merge", col='green', same_plot=True)
# plot_neighbours_from_saved_log("method1024x2000", "merge", col='pink', same_plot=True)
# plot_neighbours_from_saved_log("method2048x1000", "merge", col='black', same_plot=True)
# plot_neighbours_from_saved_log("method4096x500", "merge", col='magenta', same_plot=False)

# plot_neighbours_from_saved_normal("method1024x2000", "method1024x2000")
# plot_neighbours_from_saved_normal("method4096x500", "method4096x500")
# plot_neighbours_from_saved_normal("method2048x1000", "method2048x1000")
# plot_neighbours_from_saved_log("method4096x500", "method4096x500")
# plot_neighbours_from_saved_log("method2048x1000", "method2048x1000")

# plg = PlanarTriangulation.random_planar_triangulation(8, 'name')
# plot_graph(plg, 'trian')
# g, d = plg.find_cubic_dual_to_triangulation()
# plot_graph(g, 'cubic', d)


