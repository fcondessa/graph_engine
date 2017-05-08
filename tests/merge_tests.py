import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import merge_graph
import unittest
import networkx as nx
import time

def em_dynamic(edge1, edge2):
    return edge1['timeseries'] == edge2['timeseries']

def em(edge1, edge2):
    return edge1['weight'] == edge2['weight']

class TestCorrectness(unittest.TestCase):
    def setUp(self):
        self.Graphs = [nx.Graph() for i in range(5)]
        self.Graphs[0].add_weighted_edges_from([('1', '0', 1.5), ('2', 'Three', 4), ('1', '2', 2)])
        self.Graphs[1].add_weighted_edges_from([('1', '2', 1.5), ('Three', '2', 2), ('0', 'Three', 2)])
        self.Graphs[2].add_weighted_edges_from([('1', '0', 1.5), ('2', 'Three', 4), ('1', '2', 2)])
        self.Graphs[4].add_weighted_edges_from([('Three', '4', 2.6), ('0', '2', 3.8), ('1', '2', 1)])
        self.result = nx.Graph()
        self.result.add_weighted_edges_from([('0', '1', 3), ('2', 'Three', 10), 
                ('1', '2', 6.5), ('0', 'Three', 2), ('Three', '4', 2.6), ('0', '2', 3.8)])

    def test_merge_all_iterative(self):
        result = merge_graph.merge_graphs_all(self.Graphs)
        self.assertTrue(nx.is_isomorphic(result, self.result, edge_match = em))

    def test_merge_all_parallel(self):
        result = merge_graph.merge_graphs_all_parallel(self.Graphs)
        self.assertTrue(nx.is_isomorphic(result, self.result, edge_match = em))

    def test_merge_sparse_dict(self):
        result = merge_graph.merge_graphs_sparse_dict(self.Graphs)
        self.assertTrue(nx.is_isomorphic(result, self.result, edge_match = em))

    def test_merge_sparse_dict_custom(self):
        result = merge_graph.merge_graphs_sparse_dict_custom(self.Graphs)
        self.assertTrue(nx.is_isomorphic(result, self.result, edge_match = em))

class TestDynamicGraphCorrectness(unittest.TestCase):
    def setUp(self):
        import dynamicGraph
        self.dyG = dynamicGraph 
        self.Graphs = [nx.Graph() for i in range(5)]
        self.Graphs[0].add_edges_from([
            ('1', '0', {'timeseries': [(0, 6, [4]), (4, 10, [1, 2, 3, 4])]}), 
            ('2', 'Three', {'weight': 4}), ('1', '2', {'weight': 2})
        ])
        self.Graphs[1].add_edges_from([
            ('1', '2', {'timeseries': [(-1, 5, [3, 2])]}),
            ('Three', '2', {'timeseries': [(4, 6, [6, 9, 4, 2])]}),
            ('0', 'Three', {'weight': 2})
        ])
        self.Graphs[2].add_edges_from([
            ('1', '0', {'weight':1.5}),
            ('2', 'Three', {'timeseries': [(3, 8, [4, 5, 6, 7])]}),
            ('1', '2', {'timeseries': [(None, 15, [1]), (15, 19, [2]), 
                (19, 25, [3, 4])]})
        ])
        self.Graphs[4].add_edges_from([
            ('Three', '4', {'timeseries': [(-4, 8, [4]), (1, 5, [0.2, 0.3])]}),
            ('0', '2', {'timeseries': [(None, 15, [2]), (14, 20, [1, 2, 3])]}),
            ('1', '2', {'timeseries': [(2, 6, [0.6, 0.8, 1])]})
        ])
        self.Graphs = [self.dyG.DynamicGraph(Graph) for Graph in self.Graphs]
        self.result = self.dyG.DynamicGraph()
        self.result.add_edges_from([
            ('1', '0', {'timeseries': [(0, 6, [4]), (4, 10, [1, 2, 3, 4]), 
                (None, None, [1.5])]}),
            ('2', 'Three', {'timeseries': [(None, None, [4]), 
                (4, 6, [6, 9, 4, 2]), (3, 8, [4, 5, 6, 7])]}),
            ('1', '2', {'timeseries': [(None, None, [2]), (-1, 5, [3, 2]), 
                (None, 15, [1]), (15, 19, [2]), (19, 25, [3, 4]), 
                (2, 6, [0.6, 0.8, 1])]}),
            ('Three', '4', {'timeseries': [(-4, 8, [4]), (1, 5, [0.2, 0.3])]}),
            ('0', '2', {'timeseries': [(None, 15, [2]), (14, 20, [1, 2, 3])]}),
            ('0', 'Three', {'timeseries': [(None, None, [2])]})
        ])

    def test_merge_all_iterative(self):
        result = merge_graph.merge_graphs_all(self.Graphs, 
            merge_fn = self.dyG.merge_fn)
        self.assertTrue(nx.is_isomorphic(result, self.result, 
            edge_match = em_dynamic))

    def test_merge_all_parallel(self):
        result = merge_graph.merge_graphs_all_parallel(self.Graphs, 
            merge_fn = self.dyG.merge_fn)
        self.assertTrue(nx.is_isomorphic(result, self.result, 
            edge_match = em_dynamic))

def timed_test(decorated_test):
    def run_test(self, *kw, **kwargs):
        start = time.time()
        decorated_test(self, *kw, **kwargs)
        end = time.time()
        print "test_duration: %s (seconds)" % (end - start)
    return run_test

class TestTimingSameGraphs(unittest.TestCase):
    def setUp(self):
        self.G = nx.read_gexf('../../samt/data/idea_map_tiny.gexf')
        print(self.G.number_of_edges())
        self.Graphs = [self.G]*1000

    @timed_test
    def test_merge_all_iterative(self):
        merge_graph.merge_graphs_all(self.Graphs)

    @timed_test
    def test_merge_all_parallel(self):
        merge_graph.merge_graphs_all_parallel(self.Graphs)

    @timed_test
    def test_merge_sparse_dict(self):
        merge_graph.merge_graphs_sparse_dict(self.Graphs)


class TestTimingDiffGraphs(unittest.TestCase):
    def setUp(self):
        from os import listdir
        from os.path import isfile, join
        mypath = '../usr_graphs/'
        files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.Graphs = list()
        for f in files:
            self.Graphs.append(nx.read_gexf(join(mypath, f)))


    @timed_test
    def test_merge_all_iterative(self):
        merge_graph.merge_graphs_all(self.Graphs)

    @timed_test
    def test_merge_all_parallel(self):
        merge_graph.merge_graphs_all_parallel(self.Graphs)

class TestTimingBigGraphs(unittest.TestCase):
    def setUp(self):
        G1_name = '../../samt/data/idea_map_small.gexf'
        G2_name = '../../samt/data/politics_map.gexf'
        G3_name = '../../samt/data/old_politics_map.gexf' 
        G1 = nx.read_gexf(G1_name)
        G2 = nx.read_gexf(G2_name)
        G3 = nx.read_gexf(G3_name)
        self.Graphs = [G1, G2, G3] * 100

    @timed_test
    def test_merge_all_iterative(self):
        merge_graph.merge_graphs_all(self.Graphs) 

    @timed_test
    def test_merge_sparse_dict(self):
        merge_graph.merge_graphs_sparse_dict(self.Graphs)


