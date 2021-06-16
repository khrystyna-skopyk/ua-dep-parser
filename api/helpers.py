from collections import defaultdict
 
class Graph():
    def __init__(self,vertices):
        self.graph_dict = defaultdict(list)
        self.vertices = vertices
 
    def append_edge(self,id,vertice):
        self.graph_dict[id].append(vertice)
 
    def is_cyclic_util(self, vertice, visited, stack):
        visited[vertice] = True
        stack[vertice] = True
 
        for neighbour in self.graph_dict[vertice]:
            if visited[neighbour] == False:
                if self.is_cyclic_util(neighbour, visited, stack) == True:
                    return True
            elif stack[neighbour] == True:
                return True
 
        stack[vertice] = False
        return False
 
    def check_is_cyclic(self):
        visited = [False] * (self.vertices + 1)
        stack = [False] * (self.vertices + 1)
        for node in range(self.vertices):
            if visited[node] == False:
                if self.is_cyclic_util(node,visited,stack) == True:
                    return True
        return False

