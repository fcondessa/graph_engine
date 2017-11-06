""" json2graph.py: graph creation engine from .json

Configurable (yaml) graph creation engine

"""

__author__ = "Filipe Condessa"
__copyright__ = "Filipe Condessa, 2017"
__maintainer__ = "Filipe Condessa"
__email__ = "fcondessa@gmail.com"
__status__ = "prototype"

import yaml
import json
import networkx as nx
from itertools import combinations
from collections import defaultdict
import os
import glob
import matplotlib.pyplot as plt

class GraphMerger:
	def __init__(self,opt=0):
		self.opt = opt
	def merge(self,G1,G2):
		if opt == 0:
			return self.merge_graphs(G1,G2)
		elif opt == 1:
			return self.merge_graphs1(G1,G2)

	def merge_graphs(self,G1, G2):
		G = nx.Graph()
		G.add_nodes_from(G1.nodes(data=True) + G2.nodes(data=True))
		B1 = G1.edges(data=True)
		C = [(elem[0], elem[1], elem[2]['attribute']) for elem in B1]
		# adding the edges
		for elem in G2.edges(data=True):
			if (elem[0], elem[1], elem[2]['attribute']) in C:
				idx = C.index((elem[0], elem[1], elem[2]['attribute']))
				B1[idx][2]['weight'] += elem[2]['weight']
			else:
				B1.append(elem)
		G.add_edges_from(B1)
		return G

	def merge_graphs1(self,G1, G2):
		G = nx.Graph()
		G.add_nodes_from(G1.nodes(data=True) + G2.nodes(data=True))
		# builds a default dict for the edges\
		C = defaultdict(int)
		# we should account for the order of the edges
		for elem in G1.edges(data=True):
			# small heuristic here to just make sure that there are no repeated edges
			d = sorted([elem[0], elem[1]])
			C[(d[0], d[1], elem[2]['attribute'])] += elem[2]['weight']
		for elem in G2.edges(data=True):
			d = sorted([elem[0], elem[1]])
			C[(elem[0], elem[1], elem[2]['attribute'])] += elem[2]['weight']
		G.add_edges_from([(elem[0], elem[1], {'attribute': elem[2], 'weight': C[elem]}) for elem in C])
		return G

class GraphBuilder:
	def __init__(self,path_spec_file):
		self.path_spec_file = path_spec_file
		with open(self.path_spec_file) as f:
			self.specifications = yaml.load(f)
		self.input_data = self.load_data()
		G = self.populate_graph(self.input_data, self.specifications)
		self.graph = self.clean_graph(G, self.specifications)

	def load_data(self):
		input_f = self.specifications['input_file']
		data = []
		# single file reader
		if os.path.isfile(input_f):
			with open(input_f, 'r') as f:
				try:
					data = json.load(f)
				except:
					with open(input_f, 'r') as f:
						print 'multiline json'
						for line in f.readlines():
							data += [json.loads(line)]
		# folder reader
		else:
			for file in glob.glob(os.path.join(input_f, '*.json')):
				with open(file, 'r') as f:
					try:
						data += json.load(f)
					except:
						with open(file, 'r') as f:
							for line in f.readlines():
								data += [json.loads(line)]
		return data

	def populate_graph(self,input_data,specifications):
		G = nx.Graph()
		# edge_dict = defaultdict(int)
		# edge_info = []
		# node_vec = []
		i = 0
		# ntot = len(input_data)
		for data_elem in input_data:
			i += 1
			try:
				elem_node_info = []
				for node_type1 in specifications['nodes']:
					node_type = specifications['nodes'][node_type1]['origin']
					node = data_elem
					if not isinstance(node_type, list):
						ntype = [node_type]
					else:
						ntype = node_type

					nax = self.get_nodes(data_elem, ntype)
					for elem in nax:
						if elem != None:
							G.add_node(elem)
							elem_node_info += [(elem, node_type1)]
				for edge_type in specifications['edges']:
					edge_spec = specifications['edges'][edge_type]
					candidate_edges = self.get_edges(elem_node_info, edge_spec)
					for edge in candidate_edges:
						try:
							G[edge[0]][edge[1]]['weight'] += edge[2]
						except:
							G.add_edge(edge[0], edge[1], weight=edge[2], attribute=edge_type)
						# print 'new_edge'
			except:
				1
		return G

	def get_nodes(self,data_elem, node_spec, nodes=[]):
		if len(node_spec) == 0:
			return []
		elif len(node_spec) == 1:
			if isinstance(data_elem, list):
				ras = []
				for elem in data_elem:
					ras += self.get_nodes(elem, node_spec, nodes)

				return ras

			else:
				auxa = data_elem[node_spec[0]]
				if isinstance(auxa, int):
					auxa = str(auxa)
				return [auxa]
		else:
			auxa = data_elem[node_spec[0]]
			if isinstance(auxa, list):
				ras = []
				for elem in auxa:
					ras += self.get_nodes(elem, node_spec[1:], nodes)
				return nodes + ras
			else:
				return nodes + self.get_nodes(auxa, node_spec[1:], nodes)

	def get_edges(self,elem_node, edge_spec):
		targets = edge_spec['connection']
		edges = []
		if edge_spec['condition'] == 'if exists':
			candidates_1 = [elem for elem in elem_node if elem[1] == edge_spec['connection'][0]]
			candidates_2 = [elem for elem in elem_node if elem[1] == edge_spec['connection'][1]]
			if len(candidates_1) > 0 and len(candidates_2) > 0:
				edges = [(elem1[0], elem2[0], edge_spec['weight']) for elem1 in candidates_1 for elem2 in candidates_2]
		if edge_spec['condition'] == 'all':
			# not accounting directed edges for now
			candidates = [elem for elem in elem_node if elem[1] in edge_spec['connection']]
			for pair in combinations(candidates, 2):
				edges += [(pair[0][0], pair[1][0], edge_spec['weight'])]
			return edges
		return edges

	def clean_graph(self,G, specifications):
		try:
			for elem in specifications['clean']['nodeid']:
				try:
					G.remove_node(elem)
				except:
					1
				# this accounts for nonexisting node id
			try:
				G.remove_node(None)
			except:
				1
			# this accounts for the fact that None can exist as a node name
		except:
			0
		# cleaning the graph
		order = []
		for elem in specifications['clean']:
			if elem == 'node_id':
				1
			else:
				order.append((elem, specifications['clean'][elem]['priority']))
		vals = [i[1] for i in order]
		priority_list = [order[i[0]][0] for i in sorted(enumerate(vals), key=lambda x: x[1])]

		for elem in priority_list:
			if elem == 'node_id':
				1

			else:
				G = self.clean_graph_unitary(G, specifications['clean'][elem]['condition'])
		return G

	def clean_graph_unitary(self,G, specifications):
		target = specifications[0]
		characteristic = specifications[1]
		if target == 'nodes':
			if characteristic == 'degree':
				degree = G.degree(G.nodes())
				if specifications[2] == 'leq':
					for elem in degree.keys():
						if degree[elem] <= int(specifications[3]):
							G.remove_node(elem)
				if specifications[2] == 'geq':
					for elem in degree.keys():
						if degree[elem] >= int(specifications[3]):
							G.remove_node(elem)
				if specifications[2] == 'l':
					for elem in degree.keys():
						if degree[elem] < int(specifications[3]):
							G.remove_node(elem)
				if specifications[2] == 'g':
					for elem in degree.keys():
						if degree[elem] > int(specifications[3]):
							G.remove_node(elem)
		if target == 'edges':
			if characteristic == 'weight':
				edges = G.edges()
				if specifications[3] == 'leq':
					for edge in edges:
						if (G[edge[0]][edge[1]]['attribute'] == specifications[2]) and (
									G[edge[0]][edge[1]]['weight'] <= int(specifications[4])):
							G.remove_edge(edge[0], edge[1])
				if specifications[3] == 'l':
					for edge in edges:
						if (G[edge[0]][edge[1]]['attribute'] == specifications[2]) and (
									G[edge[0]][edge[1]]['weight'] < int(specifications[4])):
							G.remove_edge(edge[0], edge[1])
				if specifications[3] == 'geq':
					for edge in edges:
						if (G[edge[0]][edge[1]]['attribute'] == specifications[2]) and (
									G[edge[0]][edge[1]]['weight'] >= int(specifications[4])):
							G.remove_edge(edge[0], edge[1])
				if specifications[3] == 'g':
					for edge in edges:
						if (G[edge[0]][edge[1]]['attribute'] == specifications[2]) and (
									G[edge[0]][edge[1]]['weight'] > int(specifications[4])):
							G.remove_edge(edge[0], edge[1])

		return G

	def write_to_file(self,output_path=''):
		if output_path == '':
			nx.write_gexf(self.Graph, self.specifications['output_file'])
		else:
			nx.write_gexf(self.Graph, output_path)

if __name__ == "__main__":
	# toy example with connecting users to messages
	path_spec_file = os.path.join('configurations','level5.yaml')
	GB = GraphBuilder(path_spec_file)
	G = GB.graph
	G.write_gexf()
