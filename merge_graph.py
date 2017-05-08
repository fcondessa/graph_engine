import networkx as nx
from collections import defaultdict
import time
from multiprocessing import Process, Manager, Lock, Pool
from multiprocessing.managers import BaseManager
import itertools
import numpy as np
import scipy.sparse

def merge_graphs1(G1,G2):
    G = nx.Graph()
    G.add_nodes_from(G1.nodes(data=True)+G2.nodes(data=True))
    # builds a default dict for the edges\
    C = defaultdict(int)
    # we should account for the order of the edges
    for elem in G1.edges(data=True):
        # small heuristic here to just make sure that there are no repeated edges
        d = sorted([elem[0],elem[1]])
        C[(d[0],d[1],elem[2]['weight'])] += elem[2]['weight']
    for elem in G2.edges(data=True):
        d = sorted([elem[0],elem[1]])
        C[(elem[0],elem[1],elem[2]['weight'])] += elem[2]['weight']
    G.add_edges_from([(elem[0],elem[1],{'weight':C[elem]}) for elem in C])
    return G

'''
    merge_graphs_iterative:
        G1 : networkx Graph
        G2 : networkx Graph
        destructive: optional argument. If true, the result overwrites G1, 
            otherwise the result is stored in a new Graph

    Returns: 
        G : Graph with nodes and edges of both graphs, with the weights of 
            common edges summed up
'''

# def merge_graphs_iterative(G1, G2, destructive = False):
#     if (destructive): G = G1
#     else: 
#         G = nx.Graph()
#         G.add_nodes_from(G1.nodes_iter(data=True))
#         G.add_edges_from(G1.edges_iter(data=True))
#     G.add_nodes_from(G2.nodes_iter(data=True))
#     for edge in G2.edges_iter(data=True):
#         (u, v, weight) = (edge[0], edge[1], edge[2]['weight'])
#         if G.has_edge(u, v):
#             G[u][v]['weight'] += weight
#         else:
#             G.add_edge(u, v, edge[2])
#     return G

def merge_fn_nx_graph(G, edge):
    (u, v, weight) = (edge[0], edge[1], edge[2]['weight'])
    if G.has_edge(u, v):
        G[u][v]['weight'] += weight
    else:
        G.add_edge(u, v, edge[2])

def merge_graphs_iterative(G1, G2, merge_fn = merge_fn_nx_graph, destructive = False):
    if (destructive): G = G1
    else: 
        G = nx.Graph()
        G.add_nodes_from(G1.nodes_iter(data=True))
        G.add_edges_from(G1.edges_iter(data=True))
    G.add_nodes_from(G2.nodes_iter(data=True))
    for edge in G2.edges_iter(data=True):
        merge_fn(G, edge)
    return G


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x

def merge_graphs_all(Graphs, merge_fn = merge_fn_nx_graph):
    G = nx.Graph()
    for graph in Graphs:
        merge_graphs_iterative(G, graph, merge_fn = merge_fn, destructive = True)
    return G

def merge_all_helper(tup):
    return merge_graphs_all(tup[0], tup[1])

def merge_graphs_all_parallel(Graphs, merge_fn = merge_fn_nx_graph, c = 1):
    p = Pool()
    node_divisor = int(len(p._pool)*c)
    node_chunks = list(chunks(Graphs, max(2, int(len(Graphs)/node_divisor))))
    temp = p.map(merge_all_helper, zip(node_chunks, [merge_fn] * node_divisor))
    result = merge_graphs_all(temp, merge_fn = merge_fn)
    return result

def s2(tup):
    (Graphs, nodes_dict) = tup
    l = len(nodes_dict)
    A = scipy.sparse.csr_matrix((l, l))
    for graph in Graphs:
        len_of_edges = graph.number_of_edges()
        (data, row_ind, col_ind) = (np.zeros(len_of_edges), 
            np.zeros(len_of_edges), np.zeros(len_of_edges))
        for i, edge in enumerate(graph.edges_iter(data = True)):
            weight = edge[2]['weight']
            zero = nodes_dict[edge[0]]; one = nodes_dict[edge[1]]
            data[i] = weight 
            row_ind[i] = zero 
            col_ind[i] = one 
        csr_mat = scipy.sparse.coo_matrix((data, (row_ind, col_ind)), shape = (l, l)).tocsr()
        A = A + csr_mat
    return A

def scipy_to_nx_graph(A, mapping, edge_attribute = 'weight'):
    G = nx.Graph()
    for i,j,d in zip(A.row, A.col, A.data):
        G.add_edge(mapping[i], mapping[j], **{edge_attribute:d})
    return G

def merge_graphs_sparse_dict_custom(Graphs):
    p = Pool()
    nodes = set()
    for graph in Graphs:
        nodes = nodes.union(set(graph.nodes()))
    l = len(nodes)
    A = scipy.sparse.csr_matrix((l, l))
    nodes_list = list(nodes)
    nodes_dict = dict()
    for i, n in enumerate(nodes_list):
        nodes_dict[n] = i
    node_divisor = int(len(p._pool))
    node_chunks = list(chunks(Graphs, max(2, int(len(Graphs)/node_divisor))))
    temp = p.map(s2, zip(node_chunks, [nodes_dict]*4))
    for sparse_mat in temp:
        A = A + sparse_mat
    G = scipy_to_nx_graph(A.tocoo(), mapping = {v: k for k, v in nodes_dict.iteritems()})
    return G

def merge_graphs_sparse_dict(Graphs):
    p = Pool()
    nodes = set()
    for graph in Graphs:
        nodes = nodes.union(set(graph.nodes()))
    l = len(nodes)
    A = scipy.sparse.csr_matrix((l, l))
    nodes_list = list(nodes)
    nodes_dict = dict()
    for i, n in enumerate(nodes_list):
        nodes_dict[n] = i
    node_divisor = int(len(p._pool))
    node_chunks = list(chunks(Graphs, max(2, int(len(Graphs)/node_divisor))))
    temp = p.map(s2, zip(node_chunks, [nodes_dict]*4))
    for sparse_mat in temp:
        A = A + sparse_mat
    G = nx.from_scipy_sparse_matrix(A.tocoo()) 
    G = nx.relabel_nodes(G, mapping = {v: k for k, v in nodes_dict.iteritems()})
    return G

# def s(tup):
#     (Graphs, nodes_list) = tup
#     l = len(nodes_list)
#     A = scipy.sparse.csr_matrix((l, l))
#     for graph in Graphs:
#         csr_mat = nx.to_scipy_sparse_matrix(graph, nodelist = nodes_list, format = 'coo')
#         A = A + csr_mat
#     return A

# def merge_graphs_sparse(Graphs):
#     p = Pool()
#     nodes = set()
#     for graph in Graphs:
#         nodes = nodes.union(set(graph.nodes()))
#     l = len(nodes)
#     A = scipy.sparse.csr_matrix((l, l))
#     nodes_list = list(nodes)
#     node_divisor = int(len(p._pool))
#     node_chunks = list(chunks(Graphs, max(2, int(len(Graphs)/node_divisor))))
#     temp = p.map(s, zip(node_chunks, [nodes_list]*4))
#     for sparse_mat in temp:
#         A = A + sparse_mat
#     return nx.from_scipy_sparse_matrix(A)

# def s1(tup):
#     (sparse_graphs, nodes_list) = tup
#     l = len(nodes_list)
#     A = scipy.sparse.csr_matrix((l, l))
#     for graph in sparse_graphs:
#         A = A + graph
#     return A

# def merge_graphs_sparse_precomp(sparse_graphs, nodes_list):
#     p = Pool()
#     l = len(nodes_list)
#     A = scipy.sparse.csr_matrix((l, l))
#     node_divisor = int(len(p._pool))
#     node_chunks = list(chunks(sparse_graphs, max(2, int(len(Graphs)/node_divisor))))
#     temp = p.map(s1, zip(node_chunks, [nodes_list]*4))
#     for sparse_mat in temp:
#         A = A + sparse_mat
#     return nx.from_scipy_sparse_matrix(A)

# def add_first_graph_edges(G, edges_iter):
#     G.add_edges_from(edges_iter)

# def add_second_graph_edges(G, edges_iter):
#     for edge in edges_iter:
#         (u, v, weight) = (edge[0], edge[1], edge[2]['weight'])
#         d = edge[2]
#         if G.has_edge(u, v):
#             d['weight'] += G.get_edge_data(u, v)['weight']
#             G.add_edge(u, v, d)
#         else:
#             G.add_edge(u, v, edge[2])

# doesn't work well because needs to generate lists of edges
# def merge_graphs_parallel(G1, G2):
#     H = nx.Graph()
#     H.add_nodes_from(G1.nodes_iter(data=True))
#     H.add_nodes_from(G2.nodes_iter(data=True))
    
#     BaseManager.register('Graph', nx.Graph)
#     manager = BaseManager()
#     manager.start()
#     G = manager.Graph(H)    
#     node_divisor = 2
#     G1_chunks = chunks(G1.edges_iter(data=True), int(G1.number_of_edges()/node_divisor))
#     # num_chunks = len(node_chunks)
#     jobs = []
#     # for edge in G1.edges_iter():
#     #     print edge
#     for chunk in G1_chunks:
#         p = Process(target = add_first_graph_edges, args = (G, chunk))
#         p.start()
#         jobs.append(p)
#     for job in jobs:
#         job.join()

#     jobs = []
#     G2_chunks = chunks(G2.edges_iter(data=True), int(G2.number_of_edges()/node_divisor))

#     for chunk in G2_chunks:
#         p = Process(target = add_second_graph_edges, args = (G, chunk))
#         p.start()
#         jobs.append(p)
#     for job in jobs:
#         job.join()

#     return G
# G = nx.Graph()
# G.add_weighted_edges_from([(0, 1, 2.5), (1, 2, 6), (3, 4, 0.8)])
# H = nx.Graph()
# H.add_weighted_edges_from([(3, 2, 5) , (0, 1, 4)])

# merge function with locking
# def merge_graphs_lists(G1, G2, l, destructive = False):
#     if (destructive): 
#         G = G1
#     else: 
#         l.acquire()
#         G = nx.Graph()
#         G.add_nodes_from(G1.nodes(data=True))
#         G.add_edges_from(G1.edges(data=True))
#         l.release()
#     l.acquire()
#     G.add_nodes_from(G2.nodes(data=True))
#     l.release()
#     for edge in G2.edges(data=True):
#         (u, v, d) = (edge[0], edge[1], edge[2])
#         if G.has_edge(u, v):
#             l.acquire()
#             d['weight'] += G.get_edge_data(u, v)['weight']
#             G.add_edge(u, v, d)
#             l.release()
#         else:
#             l.acquire()
#             G.add_edge(u, v, edge[2])
#             l.release()
#     if (not destructive): return G



# def h(chunk):
#     return merge_graphs_all(chunk)

# def chunks_2(l, n):
#     """Divide a list of nodes `l` in `n` chunks"""
#     l_c = iter(l)
#     while 1:
#         x = tuple(itertools.islice(l_c, n))
#         if not x:
#             return
#         yield x

# def h(G, chunk, l):
#     for graph in chunk:
#         merge_graphs_lists(G, graph, l, destructive = True)


# send edges in parallel, locking and unlocking causing bottleneck
# def merge_graphs_all_parallel(Graphs, l):
#     BaseManager.register('Graph', nx.Graph)
#     manager = BaseManager()
#     manager.start()
#     G = manager.Graph()

#     chunks_list = list(chunks_2(Graphs, 10))
#     # p.map(h, zip([G]*len(chunks_list), chunks_list))
#     jobs = []
#     for chunk in chunks_list:
#         p = Process(target = h, args = (G, chunk, l))
#         p.start()
#         jobs.append(p)
#     for job in jobs:
#         job.join()
#     return G

# def merge_graphs_all(Graphs):
#   G = nx.Graph()

# G1_name = '../samt/data/idea_map_small.gexf'
# G2_name = '../samt/data/idea_map.gexf'
# G3_name = '../samt/data/idea_map_tiny.gexf'

# # G_name = 'output_graph/merged_graph.gexf'
# # G1 = nx.read_gexf(G1_name)
# # G2 = nx.read_gexf(G2_name)
# G3 = nx.read_gexf(G3_name)
# # G1 = nx.Graph()
# # G1.add_weighted_edges_from([('1', '0', 1.5), ('2', 'Three', 4), ('1', '2', 2)])
# # G2 = nx.Graph()
# # G2.add_weighted_edges_from([('1', '2', 1.5), ('Three', '2', 2), ('0', 'Three', 2)])
# # G3 = nx.Graph()
# # G3.add_weighted_edges_from([('1', '0', 1.5), ('2', 'Three', 4), ('1', '2', 2)])
# # G = merge_graphs_sparse_dict([G1, G2, G3])
# # Gd = merge_graphs_sparse_dict([G1, G2, G3])

# start_ = time.time()
# for i in range(1):
#     G = merge_graphs_all([G3]*100)
# end_ = time.time()
# print("iterative:    %f" % (end_ - start_))
# start = time.time()
# for i in range(1):
#     G = merge_graphs_all_parallel([G3]*1000, 1)
# end = time.time()
# print("parallelized: %f" % (end - start))
# start_sp = time.time()
# # for i in range(1):
# #     G = merge_graphs_sparse([G3]*1000)
# # end_sp = time.time()
# # print("sparse:    %f" % (end_sp - start_sp))


# # Graphs = [G3]*1000
# # nodes = set()
# # for graph in Graphs:
# #     nodes = nodes.union(set(graph.nodes()))
# # nodes_list = list(nodes)
# # s_g = [nx.to_scipy_sparse_matrix(graph) for graph in Graphs]
# # start_sp = time.time()
# # for i in range(1):
# #     G = merge_graphs_sparse_precomp(s_g, nodes_list)
# # end_sp = time.time()
# # print("sparse_precomputed:    %f" % (end_sp - start_sp))

# start_di = time.time()
# for i in range(1):
#     G = merge_graphs_sparse_dict([G3]*1000)
# end_di = time.time()
# print("sparse_dict:    %f" % (end_di - start_di))
# start_di = time.time()

# tart_di = time.time()
# for i in range(1):
#     G = merge_graphs_sparse_dict_custom([G3]*1000)
# end_di = time.time()
# print("sparse_dict_custom:    %f" % (end_di - start_di))
# start_di = time.time()

# # nx.write_gexf(G,G_name)


