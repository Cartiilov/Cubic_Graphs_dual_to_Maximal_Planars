from __future__ import annotations
from Graph import Graph
import numpy as np
import random
import networkx as nx
import queue as q
import time



class PlanarTriangulation():
    def __init__(self, area_matrix: np.ndarray):
        self._area_matrix = area_matrix.copy()

    def to_adjacency_matrix(self) -> np.ndarray:
        return PlanarTriangulation.area_matrix_to_adjacency_matrix(self._area_matrix)

    def add_node_to_cubic(self, vertex1, vertex2, cubic: np.ndarray) -> tuple(np.ndarray, np.ndarray):
        area_matrix = self._area_matrix
        i1 = np.where(np.any(area_matrix == vertex1, axis=1))
        i2 = np.where(np.any(area_matrix == vertex2, axis=1))
        # common edge is the indexes of the rows in area_matrix that have a common edge
        common_edge = set(i1[0]).intersection(i2[0])
        if len(common_edge) == 1:
            return cubic
        node1, node2 = common_edge.pop(), common_edge.pop()
        cubic[node1][node2] = cubic[node2][node1] = 1
        return cubic

    def find_cubic_dual_to_triangulation(self) -> tuple(Graph, dict):
        area_matrix = self._area_matrix
        c_nodes = area_matrix.shape[0] + 1
        cubic_adj_m = np.zeros((c_nodes, c_nodes), dtype=np.int64)

        mapped_areas = {tuple(row): False for row in area_matrix}
        for key, item in mapped_areas.items():
            cubic_adj_m = self.add_node_to_cubic(key[0], key[1], cubic_adj_m)
            cubic_adj_m = self.add_node_to_cubic(key[1], key[2], cubic_adj_m)
            cubic_adj_m = self.add_node_to_cubic(key[2], key[0], cubic_adj_m)

        conn = 0
        for i in range(cubic_adj_m.shape[0]):
            for j in range(cubic_adj_m.shape[1]):
                if cubic_adj_m[i][j] == 1:
                    conn += 1
            if conn < 3:
                cubic_adj_m[i][-1] = cubic_adj_m[-1][i] = 1
            conn = 0
        labels = {}
        for i in range(area_matrix.shape[0]):
            lb = "[ "
            for j in range(area_matrix.shape[1]):
                lb += str(area_matrix[i][j]) + " "
            lb += "]"
            labels[i] = lb
        labels[area_matrix.shape[0]] = " [ out ] "
        return Graph(cubic_adj_m), labels

    def insert_vertex(area_matrix: np.ndarray) -> np.ndarray:
        areas = area_matrix.shape[0]
        vertices = int((areas + 5)/2)
        insertion = random.randint(0, areas-1)
        vertex1, vertex2, vertex3 = area_matrix[insertion][0], area_matrix[insertion][1], area_matrix[insertion][2]

        area_matrix[insertion] = [vertex1, vertex2, vertices]
        area_matrix = np.append(
            area_matrix, [[vertex1, vertex3, vertices]], axis=0)
        area_matrix = np.append(
            area_matrix, [[vertex2, vertex3, vertices]], axis=0)

        return area_matrix

    @staticmethod
    def grow_triangulation(area_matrix: np.ndarray, vertices = None) -> np.ndarray:
        vertices -= 4
        if vertices is None:
            vertices = random.randint(20, 100) - 4
        for i in range(vertices):
            area_matrix = PlanarTriangulation.insert_vertex(area_matrix)

        return area_matrix

    def randomize_edge(self, area_matrix: np.ndarray, area_num, adj_matrix: np.ndarray) -> tuple(np.ndarray, bool):
        
        row1 = random.randint(0, area_matrix.shape[0]-1)
        random_area = area_matrix[row1]
        tmp = random.randint(0, 2)
        vertex1 = random_area[tmp]
        if tmp == 2:
            vertex2 = random_area[0]
        else:
            vertex2 = random_area[tmp+1]
        i1 = np.where(np.any(area_matrix == vertex1, axis=1)) 
        i2 = np.where(np.any(area_matrix == vertex2, axis=1))
        common_edge = set(i1[0]).intersection(i2[0])
        if len(common_edge) == 0:
            print("fail")
            return area_matrix, False

        row2 = common_edge.pop()
        elem1 = np.setdiff1d(random_area, [vertex1, vertex2])[0]
        elem2 = np.setdiff1d(area_matrix[row2], [vertex1, vertex2])[0]
        if not(adj_matrix[elem1][elem2] == 0 and elem1 != elem2):
            return area_matrix, False
    
        area1 = np.array([elem1, elem2, vertex1])
        area2 = np.array([elem1, elem2, vertex2])
        area1.sort()
        area2.sort()
        area_matrix[row1] = area1
        area_matrix[row2] = area2

        return area_matrix, True

    def randomize_edge2(self, area_matrix: np.ndarray, num_of_nodes, adj_matrix: np.ndarray) -> tuple(np.ndarray, bool):
        vertex1 = np.random.randint(0, num_of_nodes)
        neighbours = np.where(adj_matrix[vertex1] == 1)[0]
        vertex2 = np.random.choice(neighbours)

        i1 = np.where(np.any(area_matrix == vertex1, axis = 1))
        i2 = np.where(np.any(area_matrix == vertex2, axis = 1))
        common_edge = np.intersect1d(i1, i2)

        if len(common_edge) < 2:
            return area_matrix, False
        
        area_ids = np.random.choice(common_edge, size=2, replace=False)
        reloc1 = np.setdiff1d(area_matrix[area_ids[0]], [vertex1, vertex2])[0]
        reloc2 = np.setdiff1d(area_matrix[area_ids[1]], [vertex1, vertex2])[0]
        
        if not(adj_matrix[reloc1][reloc2] == 0 and reloc1 != reloc2):
            return area_matrix, False
        
        area1 = np.array([reloc1, reloc2, vertex1])
        area2 = np.array([reloc1, reloc2, vertex2])
        area1.sort()
        area2.sort()
        area_matrix[area_ids[0]] = area1
        area_matrix[area_ids[1]] = area2

        return area_matrix, True
        
        
    def sweep(self) -> PlanarTriangulation:
        area_m = self._area_matrix
        adj_m = PlanarTriangulation.area_matrix_to_adjacency_matrix(
            area_m)
        nodes_num = adj_m.shape[0]
        areas = area_m.shape[0]
        edges = 3*int((areas + 5)/2) - 6
        was_changed = False
        if nodes_num < 5:
            print("Cannnot randomise a graph with less than 5 vertices!")
            raise ValueError()
        for i in range(edges):
            area_m, was_changed = self.randomize_edge(area_m, nodes_num, adj_m)
            
            if was_changed:
                adj_m = PlanarTriangulation.area_matrix_to_adjacency_matrix(
                area_m)
        return PlanarTriangulation(area_m)
        
    @staticmethod  
    def sweep_from_file(graph_pth_in, graph_pth_out) -> None:
        with open('data/graphs/'+ graph_pth_in +'.npy', 'rb') as f:
            area_m = np.load(f)
        plg = PlanarTriangulation(area_m)
        plg.sweep()
        with open('data/graphs/'+ graph_pth_out +'.npy', 'wb') as f:
                np.save(f, plg._area_matrix)
        return plg
                
    @staticmethod  
    def find_cubic_from_file(graph_pth_in):
        with open('data/graphs/'+ graph_pth_in +'.npy', 'rb') as f:
            area_m = np.load(f)
        plg = PlanarTriangulation(area_m)
        g, t = plg.find_cubic_dual_to_triangulation()
        return g, t
        
        
    @staticmethod
    def area_matrix_to_adjacency_matrix(area_matrix: np.ndarray) -> np.ndarray:
        areas = area_matrix.shape[0]
        vertices = int((areas + 5)/2)
        adjacency_matrix = np.zeros((vertices, vertices))

        prev = -1

        for i in range(areas):
            for j, elem in np.ndenumerate(area_matrix[i]):
                if prev != -1:
                    adjacency_matrix[elem][prev] = adjacency_matrix[prev][elem] = 1
                prev = elem
            adjacency_matrix[area_matrix[i][0]
                             ][prev] = adjacency_matrix[prev][area_matrix[i][0]] = 1
            prev = -1
        return adjacency_matrix

    @staticmethod
    def random_planar_triangulation(number_f_nodes, save_name: str | None = None) -> PlanarTriangulation:
        
        area_matrix = np.array([[0, 1, 2],
                                [0, 2, 3],
                                [1, 2, 3]])
        
        area_matrix = PlanarTriangulation.grow_triangulation(
            area_matrix, number_f_nodes)
        if save_name:
            with open('data/graphs/'+ save_name +'.npy', 'wb') as f:
                np.save(f, area_matrix)
        return PlanarTriangulation(area_matrix)

    def number_of_neighbours(self):
        adj_m = self.to_adjacency_matrix()
        neigh_num = np.zeros(adj_m.shape[0], np.intc)
        for i in range(adj_m.shape[0]):
            for j in range(adj_m.shape[0]):
                neigh_num[i] += adj_m[i][j]
        return neigh_num
    
        
    def graph_BFS(self, histogram: np.ndarray, num_of_nodes) -> np.ndarray:
        # randomlist = random.sample(range(0, num_of_nodes-1), int(0.05*num_of_nodes))
        adj_matrix = self.to_adjacency_matrix()
        num_of_nodes = adj_matrix.shape[0]
         #find dual graph
        g = Graph(adj_matrix)

        adj_list = g.to_adjacency_list() # find adjacency list for graph
        for i in range(num_of_nodes-1): # call BFS on every node
            # histogram ->BFS works on one histogram for all iterations
            histogram = PlanarTriangulation.BFS(i, num_of_nodes, adj_list, histogram) 

        return histogram
  
    #BFS accepts graph, number of node on which it wil start, adjacency list and histogram
    #histogram stores the data of how many nodes are on each level
    @staticmethod
    def BFS(node, num_of_nodes, adj_list: np.ndarray, histogram: np.ndarray) -> np.ndarray:
        level = np.zeros(num_of_nodes, np.int32)
        marked = np.full(num_of_nodes, False)
        
        que = q.Queue()
        que.put(node)

        level[node] = 0
        marked[node] = True
        
        while (not que.empty()):
            
            node = que.get()
            
            adj_l = adj_list[node]
            for i in range(len(adj_l)):

                b = adj_l[i]

                if (not marked[b]):
                    
                    que.put(b)
                    #establish on which level the node is
                    level[b] = level[node] + 1
                    
                    #add one to the n8umber of occurences for a ceartain distance
                    histogram[level[b]] += 1
                    
                    marked[b] = True
        return histogram