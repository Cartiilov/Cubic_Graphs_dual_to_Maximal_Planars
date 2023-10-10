from __future__ import annotations
from Graph import Graph
import numpy as np
import random
import networkx as nx


class PlanarTriangulation(Graph):
    def __init__(self, adjacency_matrix: np.ndarray, area_matrix: np.ndarray):
        self._adjacency_matrix = adjacency_matrix.copy()
        self._area_matrix = area_matrix.copy()

    def insert_vertex(area_matrix: np.ndarray) -> np.ndarray:
        areas = area_matrix.shape[0]
        vertices =  int((areas + 5)/2)
        insertion = random.randint(0, areas-1)
        print("Inserting: ", insertion)
        vertex1, vertex2, vertex3 = area_matrix[insertion][0], area_matrix[insertion][1], area_matrix[insertion][2]
        
        area_matrix[insertion] = [vertex1, vertex2, vertices]
        area_matrix = np.append(
            area_matrix, [[vertex1, vertex3, vertices]], axis=0)
        area_matrix = np.append(
            area_matrix, [[vertex2, vertex3, vertices]], axis=0)
        
        return area_matrix

    @staticmethod
    def build_triangulation(trian: PlanarTriangulation) -> PlanarTriangulation:
        area_matrix = trian._area_matrix
        vertices_to_add = 100
        for i in range(vertices_to_add):
            area_matrix = PlanarTriangulation.insert_vertex(area_matrix)
        

        return PlanarTriangulation(PlanarTriangulation.area_matrix_to_adjacency_matrix(area_matrix), area_matrix)

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
        print(adjacency_matrix)
        print(area_matrix)
        return adjacency_matrix

    @staticmethod
    def random_planar_triangulation() -> PlanarTriangulation:
        area_matrix = np.array([[0, 1, 2],
                                [0, 2, 3],
                                [1, 2, 3],])
        return PlanarTriangulation(PlanarTriangulation.area_matrix_to_adjacency_matrix(area_matrix), area_matrix)
