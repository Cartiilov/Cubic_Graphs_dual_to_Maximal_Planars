from __future__ import annotations
import numpy as np
import networkx as nx

#from lab2 import graph_degree_sequence


class Graph:
    def __init__(self, adjacency_matrix: np.ndarray) -> None:
        self._adjacency_matrix = adjacency_matrix.copy()

    @staticmethod
    # def from_degree_sequence(sequence: np.ndarray | list) -> "Graph":
    #     if isinstance(sequence, list):
    #         sequence = np.array(sequence)
    #     return Graph(graph_degree_sequence.adjacency_matrix_from_sequence(sequence))

    def to_degree_sequence(self) -> list[int]:
        g = nx.from_numpy_array(self.to_adjacency_matrix())
        return [degree for _, degree in g.degree()]

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

    @staticmethod
    def from_incidence_matrix(matrix: np.ndarray):
        collumns = [matrix[:, col_index] for col_index in range(matrix.shape[1])]
        # [[1,0,1,0], [0,1,1,0]]
        enumerated = map(lambda collumn: enumerate(collumn), collumns)
        # [[(0,1), (1,0), (2,1), (3,0)], [(0,0), (1,1), (2,1), (3,0)]]
        filtered = [
            [
                *map(
                    lambda entries: entries[0],
                    filter(lambda values: values[1] == 1, enumerated_collumn),
                )
            ]
            for enumerated_collumn in enumerated
        ]
        # [[(0,1), (2,1)], [(1,1), (2,1)]]
        adj_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
        for i, j in filtered:
            adj_matrix[i][j] = adj_matrix[j][i] = 1
        return Graph(adj_matrix)

    def to_adjacency_matrix(self) -> np.ndarray:
        return self._adjacency_matrix.copy()

    def to_incidence_matrix(self):
        indexes = np.argwhere(self._adjacency_matrix == 1)

        filtered = [el for el in indexes if el[0] < el[1]]

        inc_matrix = np.zeros((self._adjacency_matrix.shape[0], len(filtered)))
        for i in range(len(filtered)):
            inc_matrix[filtered[i][0]][i] = 1
            inc_matrix[filtered[i][1]][i] = 1

        return inc_matrix

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

    @staticmethod
    def random_graph(numOfNodes, numOfEdges):
        if (
            numOfNodes < 1
            or numOfEdges < 0
            or (numOfEdges > (numOfNodes * (numOfNodes - 1) / 2))
        ):
            raise ValueError()
        rnd_graph = np.zeros((numOfNodes, numOfNodes))
        iu = np.stack(np.triu_indices(numOfNodes, 1), axis=1)
        indicies = np.random.choice(iu.shape[0], numOfEdges, replace=False)
        selected_indexes = iu[indicies]
        for i, j in selected_indexes:
            rnd_graph[i][j] = rnd_graph[j][i] = 1
        return Graph(rnd_graph)

    @staticmethod
    def random_graph_weighted(numOfNodes, probability):
        if numOfNodes < 1 or probability < 0:
            raise ValueError()
        rnd_graph = np.zeros((numOfNodes, numOfNodes))
        for i in range(rnd_graph.shape[0]):
            for j in range(rnd_graph.shape[1]):
                if i <= j:
                    continue
                else:
                    rnd_graph[i][j] = rnd_graph[j][i] = np.random.choice(
                        [0, 1], p=[1 - probability, probability]
                    )
        return Graph(rnd_graph)

    def remove_edge(self, node_1_index: int, node_2_index: int) -> Graph:
        """Remove edge between two nodes

        Args:
            node_1_index (int): The first node index a lookep up in the adjacency matrix
            node_2_index (int): The second node index

        Returns:
            Graph: Self
        """
        self._adjacency_matrix[node_1_index][node_2_index] = self._adjacency_matrix[
            node_2_index
        ][node_1_index] = 0
        return self

    def remove_node(self, node_index: int) -> int:
        """Remove node from the graph

        Args:
            node_index (int): Node index to be removed

        Returns:
            int: The removed node index
        """
        self._adjacency_matrix = np.delete(self._adjacency_matrix, node_index, 0)
        self._adjacency_matrix = np.delete(self._adjacency_matrix, node_index, 1)

        return node_index

    def remove_nodes(self, node_indexes: np.ndarray) -> list[int]:
        """Removes multiple nodes from the graph

        Args:
            node_indexes (np.ndarray): Ndarray of node indexes to be removed

        Returns:
            Graph: Self
        """

        return [self.remove_node(node_index) for node_index in node_indexes]

    def copy(self) -> Graph:
        """Create a copy of the graph

        Returns:
            Graph: Self
        """
        return Graph(self._adjacency_matrix.copy())

    @property
    def isolated_nodes(self) -> np.ndarray:
        return np.argwhere(self._adjacency_matrix.sum(axis=1) == 0).flatten()

    def get_neighbours(self, node_index: int) -> np.ndarray:
        return np.argwhere(self._adjacency_matrix[node_index] == 1).flatten()

    @property
    def nodes(self) -> np.ndarray:
        return np.arange(self._adjacency_matrix.shape[0])
    
gra = Graph.random_graph_weighted(3, 0.5)