from Graph import Graph
from planar_graph import PlanarTriangulation

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



def plot_graph(
    graph: Graph, filepath: str | None = None, sp: Graph | None = None
):
    print(graph.to_adjacency_matrix())
    g = nx.from_numpy_array(graph.to_adjacency_matrix())
    if sp is not None:
        s = nx.from_numpy_array(sp.to_adjacency_matrix())
    #pos = nx.spring_layout(g, center=(0, 0))

    plt.figure(figsize=(15, 15))
    position= nx.planar_layout(g, scale = 1, center=(0, 0))
    nx.draw(g, pos=position)
    nx.draw_networkx_labels(g, position)


    nx.draw(g, pos=position)

    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()


plg = PlanarTriangulation.random_planar_triangulation()
plot_graph(plg, '.')
plg1 = PlanarTriangulation.build_triangulation(plg)
print(plg1._area_matrix)
plot_graph(plg1, 'plg1')


