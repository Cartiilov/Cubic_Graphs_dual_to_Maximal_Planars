from __future__ import annotations
from Graph import Graph
import numpy as np
import random
import networkx as nx
import queue as q


class PlanarTriangulation():
    def __init__(self, adjacency_matrix: np.ndarray, area_matrix: np.ndarray):
        self._adjacency_matrix = adjacency_matrix.copy()
        self._area_matrix = area_matrix.copy()

    def to_adjacency_matrix(self) -> np.ndarray:
        return self._adjacency_matrix.copy()

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
            item = True

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
            #print(lb)
            labels[i] = lb
            #print(f'area_matrix[{i}]={area_matrix[i]}')
        labels[area_matrix.shape[0]] = " [ out ] "
        #print("labels", labels)
        return Graph(cubic_adj_m), labels

    def insert_vertex(area_matrix: np.ndarray) -> np.ndarray:
        areas = area_matrix.shape[0]
        vertices = int((areas + 5)/2)
        insertion = random.randint(0, areas-1)
        #print("Inserting: ", insertion)
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

    @staticmethod
    def randomize_edge(area_matrix: np.ndarray, tries) -> np.ndarray:
        adj_matrix = PlanarTriangulation.area_matrix_to_adjacency_matrix(
            area_matrix)
        if adj_matrix.shape[0] < 5:
            print("Cannnot randomise a graph with less than 5 vertices!")
            raise ValueError()
        num_of_areas = area_matrix.shape[0]
        elem1 = elem2 = vertex1 = vertex2 = row2 = row1 = -1
        
        row1 = random.randint(0, num_of_areas-1)
        random_area = area_matrix[row1]
        tmp = random.randint(0, 2)
        vertex1 = random_area[tmp]
        if tmp == 2:
            vertex2 = random_area[0]
        else:
            vertex2 = random_area[tmp+1]
        #print(vertex1, vertex2)
        i1 = np.where(np.any(area_matrix == vertex1, axis=1))
        i2 = np.where(np.any(area_matrix == vertex2, axis=1))
        common_edge = set(i1[0]).intersection(i2[0])
        if len(common_edge) == 0:
            return area_matrix
        # common_edge.remove(0)
        row2 = common_edge.pop()
        elem1 = np.setdiff1d(area_matrix[row1], [vertex1, vertex2])[0]
        elem2 = np.setdiff1d(area_matrix[row2], [vertex1, vertex2])[0]
        if not(adj_matrix[elem1][elem2] == 0 and elem1 != elem2):
            return area_matrix

        new_area1 = np.array([elem1, elem2, vertex1])
        new_area2 = np.array([elem1, elem2, vertex2])
        new_area1.sort()
        new_area2.sort()

        area_matrix[row1] = new_area1
        area_matrix[row2] = new_area2

        return area_matrix

    def randomize_graph_edges(area_matrix: np.ndarray, num_of_rand=None) -> np.ndarray:
        if num_of_rand is None:
            num_of_rand = random.randint(10, 100)
        for i in range(num_of_rand):
            area_matrix = PlanarTriangulation.randomize_edge(area_matrix)
        return area_matrix
    
    def sweep(self) -> PlanarTriangulation:
        a_m = self._area_matrix
        areas = a_m.shape[0]
        edges = 3*int((areas + 5)/2) - 6
        for i in range(edges):
            a_m = PlanarTriangulation.randomize_edge(a_m, edges)
        return PlanarTriangulation(PlanarTriangulation.area_matrix_to_adjacency_matrix(a_m), a_m)
        

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
    def random_planar_triangulation(vertices_to_add) -> PlanarTriangulation:
        area_matrix = np.array([[0, 1, 2],
                                [0, 2, 3],
                                [1, 2, 3],])
        area_matrix = PlanarTriangulation.grow_triangulation(
            area_matrix, vertices_to_add)
        return PlanarTriangulation(PlanarTriangulation.area_matrix_to_adjacency_matrix(area_matrix), area_matrix)

    def number_of_neighbours(self):
        adj_m = self.to_adjacency_matrix()
        neigh_num = np.zeros(adj_m.shape[0], np.intc)
        #histogram = np.zeros(adj_m.shape[0])
        for i in range(adj_m.shape[0]):
            for j in range(adj_m.shape[0]):
                neigh_num[i] += adj_m[i][j]
        #bin_edges = np.round(bin_edges,0)
        return neigh_num
    
        
    def graph_BFS(self) -> np.ndarray:
        g, d = self.find_cubic_dual_to_triangulation()
        num_of_nodes = g.to_adjacency_matrix().shape[0]
        node_neigh = np.zeros(num_of_nodes, dtype=np.int64)
        for i in range(num_of_nodes):
            node_neigh = np.append(node_neigh, PlanarTriangulation.BFS(g, i, num_of_nodes))
        return node_neigh
  
    @staticmethod
    def BFS(g: Graph, node, num_of_nodes) -> np.ndarray:
        level = np.empty(num_of_nodes)
        level[:] = np.NaN
        marked = np.full(num_of_nodes, False)

        que = q.Queue()
        que.put(node)

        level[node] = 0
        marked[node] = True
        
        while (not que.empty()):
            node = que.get()
            
            adj_l = g.to_adjacency_list()[node]
            for i in range(len(adj_l)):

                b = adj_l[i]

                if (not marked[b]):

                    que.put(b)
                    level[b] = level[node] + 1
                    marked[b] = True
        
        return level