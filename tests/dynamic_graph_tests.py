import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dynamicGraph
import unittest
import networkx as nx

def em(edge1, edge2):
    return edge1['weight'] == edge2['weight']

class TestDynamicGraph(unittest.TestCase):
    def setUp(self):
        self.G = dynamicGraph.DynamicGraph()
        self.G.add_edges_from([
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

    def testGraphAtTime(self):
        H = nx.Graph()
        H.add_weighted_edges_from([('1', '0', 5.5), ('2', 'Three', 4), 
            ('1', '2', 6), ('Three', '4', 4.2), ('0', '2', 2),
            ('0', 'Three', 2)
        ])
        self.assertTrue(nx.is_isomorphic(self.G.at(1), H, edge_match = em))
