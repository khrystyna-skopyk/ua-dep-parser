from collections import defaultdict
 
class Graph():
    def __init__(self,vertices):
        self.graph = defaultdict(list)
        self.V = vertices
 
    def add_edge(self,u,v):
        self.graph[u].append(v)
 
    def is_cyclic_util(self, v, visited, recStack):
        visited[v] = True
        recStack[v] = True
 
        for neighbour in self.graph[v]:
            if visited[neighbour] == False:
                if self.is_cyclic_util(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True
 
        recStack[v] = False
        return False
 
    def check_is_cyclic(self):
        visited = [False] * (self.V + 1)
        recStack = [False] * (self.V + 1)
        for node in range(self.V):
            if visited[node] == False:
                if self.is_cyclic_util(node,visited,recStack) == True:
                    return True
        return False



