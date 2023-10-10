#from Graph import Graph
import numpy as np

#gra = Graph.random_graph_weighted(3, 0.5)

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        print(f'arr[{i}][{j}] = {arr[i][j]}')

