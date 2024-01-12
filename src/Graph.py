from __future__ import annotations
import numpy as np
import networkx as nx

class Graph:
    def __init__(self, adjacency_matrix: np.ndarray) -> None:
        self._adjacency_matrix = adjacency_matrix.copy()

    @staticmethod
    def from_adjacency_matrix(matrix: np.ndarray) -> Graph:
        return Graph(matrix)

    @staticmethod
    def from_adjacency_list(adj_list: tuple[tuple[int]]):
        length = len(adj_list)
        adj_matrix = np.zeros((length, length), dtype=np.int64)

        for node_index, adj in enumerate(adj_list):
            for neighbour in adj:
                adj_matrix[node_index, neighbour] = 1
        return Graph(adj_matrix)

    def to_adjacency_matrix(self) -> np.ndarray:
        return self._adjacency_matrix.copy()

    def to_networkx_graph(self):
        return nx.from_numpy_array(self._adjacency_matrix)

    def to_adjacency_list(self):
        adj_list = []
        for i in range(self._adjacency_matrix.shape[0]):
            row = []
            for j in range(self._adjacency_matrix.shape[1]):
                if self._adjacency_matrix[i][j] != 0:
                    row.append(j)
            adj_list.append(row)

        return adj_list

    def copy(self) -> Graph:
        """Create a copy of the graph

        Returns:
            Graph: Self
        """
        return Graph(self._adjacency_matrix.copy())



    