import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph_engine
import unittest
import networkx as nx

class TestJsonToGraph(unittest.TestCase):
    def setUp(self):
        self.inputJson = {
            'a' : {
                'b' : [{'c' : ['d1', 'd2'], 'e1': 'f1'}, 'c1', 'c2'], 'd' : 'e'
            },
            'e' : {
                'f' : 'g', 'h' : 'i', 'd': 'a'
            }
        }

        self.result = nx.Graph()
        self.result.add_edges_from([('a', 'a(_)d'), ('a', 'a(_)b'), 
            ('a(_)b(_)0_c', 'a(_)b'), ('a(_)b(_)0_e1', 'a(_)b'), ('e', 'e(_)d'),
            ('e', 'e(_)h'), ('e', 'e(_)f')
        ])

    def testCorrectness(self):
        result = graph_engine.json_to_graph(self.inputJson)
        self.assertTrue(nx.is_isomorphic(result, self.result))



