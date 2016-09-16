import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools
import operator


def make_graph(nodes, edges):
    """
    Construct a Graph of Markev Net
    """
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    return graph

def select_central_nodes(graph, threshold):
    """
    Select Central nodes , i.e. nodes with the largest number of neighbors
    """
    nodes = graph.nodes(data=False)
    releventNodes = []
    for node in nodes:
        if len(graph.edges(node)) >= threshold :
            releventNodes.append(node)
    return releventNodes

def select_edge_to_inject(graph, searchSet, threshold):
    """
    Select Edge to inject for pruning task
    """
    releventNodes = select_central_nodes(graph, threshold)
    found = False
    edgeScores = dict()
    selectedEdge, resultingEdges = [], []
    if len(releventNodes) > 2:
        candidateEdges = list(itertools.combinations(releventNodes, 2))
        for edge in (candidateEdges):
            edge = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if edge in searchSet:
                neighbors1 = graph.neighbors(edge[0])
                neighbors2 = graph.neighbors(edge[1])
                if len(list(set(neighbors1 + neighbors2) - set(neighbors1) & set(neighbors2))) > 0:
                    currResultingEdges = [(x, y) for (x, y) in list(itertools.product(neighbors1, neighbors2)) +
                                          list(itertools.product(neighbors2, neighbors1))
                                          if x < y and ((x, y) in searchSet or (y, x) in searchSet)]
                    if (len(currResultingEdges)>0):
                        edgeScores[edge] = len(currResultingEdges)
        if len(edgeScores)>0:
            selectedEdge = max(edgeScores.iteritems(), key=operator.itemgetter(1))[0]
            found = True
            neighbors1 = graph.neighbors(selectedEdge[0])
            neighbors2 = graph.neighbors(selectedEdge[1])
            resultingEdges = list(set([(x, y) for (x, y) in
                              list(itertools.product(neighbors1, neighbors2)) +
                              list(itertools.product(neighbors2, neighbors1)) if
                              x < y and ((x, y) in searchSet or (y, x) in searchSet)]))
    return found, selectedEdge, resultingEdges

def draw_graph(graph, initial_nodes):
    """
    Draw Graph
    """
    labels = {}
    G = nx.Graph()
    # Add nodes and labels
    for node in initial_nodes:
        G.add_node(node)
        labels[node] = node
    # Add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])
    # Draw graph
    pos = nx.shell_layout(G)
    nx.draw(G, pos)
    plt.show()

def draw_two_graphs(graph1, graph2, initial_nodes):
    """
    Draw Graph
    """
    G1 = nx.Graph()
    G2 = nx.Graph()
    # Add nodes and labels
    for node in initial_nodes:
        G1.add_node(node)
        G2.add_node(node)
    # Add edges
    for edge in graph1:
        G1.add_edge(edge[0], edge[1])
    for edge in graph2:
        G2.add_edge(edge[0], edge[1])
    # Draw graph
    pos1 = nx.shell_layout(G1)
    pos2 = nx.shell_layout(G2)
    plt.figure(1)
    nx.draw(G1, pos1)
    plt.figure(2)
    nx.draw(G2, pos2)
    plt.show()