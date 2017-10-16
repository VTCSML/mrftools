import Queue
import numpy as np

def create_tree(nodes_ids):
    nodes = Queue.Queue()
    edges = []
    for id in nodes_ids:
        nodes.put(id)
    unvisited_nodes = Queue.Queue()
    root = nodes.get()
    unvisited_nodes.put(root)
    while unvisited_nodes.empty() == False:
        node = unvisited_nodes.get()
        if nodes.empty() == False:
            left_child = nodes.get()
            edges.append((node, left_child))
            unvisited_nodes.put(left_child)
        if nodes.empty() == False:
            right_child  = nodes.get()
            edges.append((node, right_child))
            unvisited_nodes.put(right_child)
    return edges

if __name__ == '__main__':
    N = 9
    nodes_ids = list()
    for i in range(0, N):
        nodes_ids.append(i)
    edges = create_tree(nodes_ids)
    print edges
        
    
        
    