# AUTHOR: filipe condessa
# fcondessa@gmail.com
# 2016.10.24
import yaml
import json
import networkx as nx
import os
import glob
import sys
from itertools import combinations
from pprint import pprint
from collections import defaultdict
import matplotlib.pyplot as plt

def load_data(input_f):
	data = []
	if os.path.isfile(input_f):
		with open(input_f,'r') as f:
			try:
				data = json.load(f)
			except:
				with open(input_f,'r') as f:
					print 'multiline json'
					for line in f.readlines():
						data += [json.loads(line)]
	# single level so far
	# extend this for recursive 
	else:
		for file in glob.glob(os.path.join(input_f,'*.json')):
			with open(file,'r') as f:
				try:
					data += json.load(f)
				except:
					with open(file,'r') as f:
						for line in f.readlines():
							data += [json.loads(line)]
	return data

def get_nodes(data_elem, node_spec,nodes=[]):
	if len(node_spec) == 0:
		return []
	elif len(node_spec) == 1:
		if isinstance(data_elem,list):
			ras = []
			for elem in data_elem:
				ras += get_nodes(elem,node_spec,nodes)

			return ras

		else:
			auxa = data_elem[node_spec[0]]
			if isinstance(auxa,int):
				auxa = str(auxa)
			return [auxa]
	else:
		auxa = data_elem[node_spec[0]]
		if isinstance(auxa,list):
			ras = []
			for elem in auxa:
				ras += get_nodes(elem,node_spec[1:],nodes)
			return nodes + ras
		else:
			return nodes + get_nodes(auxa,node_spec[1:],nodes)


def get_edges(elem_node,edge_spec):
	targets = edge_spec['connection']
	edges = []
	if edge_spec['condition'] == 'if exists':
		candidates_1 = [elem for elem in elem_node if elem[1] == edge_spec['connection'][0]]
		candidates_2 = [elem for elem in elem_node if elem[1] == edge_spec['connection'][1]]
		if len(candidates_1)>0 and len(candidates_2)>0:
			edges = [(elem1[0],elem2[0],edge_spec['weight']) for elem1 in candidates_1 for elem2 in candidates_2]
	if edge_spec['condition'] == 'all':
		# not accounting directed edges for now
		candidates = [ elem for elem in elem_node if elem[1] in edge_spec['connection']]
		for pair in combinations(candidates,2):
			edges += [(pair[0][0],pair[1][0],edge_spec['weight'])]
		return edges
	return edges

def clean_graph(G,specifications):
	try:
		for elem in specifications['clean']['node_id']:			
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
			order.append((elem,specifications['clean'][elem]['priority']))
	vals = [i[1] for i in order]
	priority_list = [order[i[0]][0] for i in sorted(enumerate(vals), key=lambda x:x[1])]
	
	for elem in priority_list:
		if elem == 'node_id':
			1

		else:
			G = clean_graph_unitary(G,specifications['clean'][elem]['condition'])
	return G


def clean_graph_unitary(G,specifications):
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
					if (G[edge[0]][edge[1]]['attribute'] == specifications[2]) and (G[edge[0]][edge[1]]['weight'] <= int(specifications[4])):
						G.remove_edge(edge[0],edge[1])
			if specifications[3] == 'l':
				for edge in edges:
					if (G[edge[0]][edge[1]]['attribute'] == specifications[2]) and (G[edge[0]][edge[1]]['weight'] < int(specifications[4])):
						G.remove_edge(edge[0],edge[1])
			if specifications[3] == 'geq':
				for edge in edges:
					if (G[edge[0]][edge[1]]['attribute'] == specifications[2]) and (G[edge[0]][edge[1]]['weight'] >= int(specifications[4])):
						G.remove_edge(edge[0],edge[1])
			if specifications[3] == 'g':
				for edge in edges:
					if (G[edge[0]][edge[1]]['attribute'] == specifications[2]) and (G[edge[0]][edge[1]]['weight'] > int(specifications[4])):
						G.remove_edge(edge[0],edge[1])





	return G

def populate_graph(input_data,specifications):
	G = nx.Graph()
	edge_dict = defaultdict(int)

	
	edge_info = []
	node_vec = []
	i = 0
	ntot = len(input_data)
	for data_elem  in input_data:
		i+=1
		try:
			elem_node_info = []
			for node_type1 in specifications['nodes']:
				node_type = specifications['nodes'][node_type1]['origin']
				node = data_elem
				if not isinstance(node_type,list):
					ntype = [node_type]
				else:
					ntype = node_type

				nax = get_nodes(data_elem,ntype)
				for elem in nax:
					if elem != None:
						G.add_node(elem)
						elem_node_info += [(elem,node_type1)]
			for edge_type in specifications['edges']:
				edge_spec = specifications['edges'][edge_type]
				candidate_edges = get_edges(elem_node_info,edge_spec)
				for edge in candidate_edges:
					try:
						G[edge[0]][edge[1]]['weight']+=edge[2]						
					except:
						G.add_edge(edge[0],edge[1],weight=edge[2],attribute=edge_type)
						# print 'new_edge'
		except:
			1
	return G
	# extend this later for multiple files
def builder(specifications):

	pprint('loading data')
	input_data = load_data(specifications['input_file'])		
	pprint('populating graph')
	G = populate_graph(input_data,specifications)
	pprint('cleaning graph')
	G = clean_graph(G,specifications)
	return G

def get_nodes_general(input_data):
	# print(input_data)
	nodes = []
	if isinstance(input_data, dict):
		for elem in input_data:
			nodes.append(elem) 
			nodes += get_nodes_general(input_data[elem])
	elif isinstance(input_data, list):
		for elem in input_data:
			nodes += get_nodes_general(elem)
	else:
		return [input_data]
	return nodes

def get_edges_general(input_data, prev = None):
	edges = []
	if isinstance(input_data, dict):
		for elem in input_data:
			if prev: 
				edges += [(prev, elem)]
			edges += get_edges_general(input_data[elem], prev = elem)
	elif isinstance(input_data, list):
		for elem in input_data:
			edges += get_edges_general(elem, prev = prev)
	else:
		return [(prev, input_data)]
	return edges



def json_to_graph(input_data):
	G = nx.Graph()
	nodes = get_nodes_general(input_data)
	edges = get_edges_general(input_data)
	G.add_nodes_from(nodes)
	print(edges)
	for edge in edges:
		try:
			G[edge[0]][edge[1]]['weight']+=1					
		except:
			G.add_edge(edge[0],edge[1],weight=1)
	return G

def get_nodes_general_alt(input_data, prefix = ""):
	# print(input_data)
	nodes = []
	if isinstance(input_data, dict):
		for elem in input_data:
			modified_elem = prefix + elem
			nodes.append(modified_elem) 
			nodes += get_nodes_general_alt(input_data[elem], prefix = modified_elem + "_")
	elif isinstance(input_data, list):
		for i, elem in enumerate(input_data):
			nodes += get_nodes_general_alt(elem, prefix = '%s%d_' % (prefix, i))
	else:
		return [input_data]
	return nodes

def get_edges_general_alt(input_data, prev = None, prefix = ""):
	edges = []
	if isinstance(input_data, dict):
		for elem in input_data:
			modified_elem = prefix + elem
			if prev: 
				edges += [(prev, modified_elem)]
			edges += get_edges_general_alt(input_data[elem], prev = modified_elem, prefix = modified_elem + "_")
	elif isinstance(input_data, list):
		for i, elem in enumerate(input_data):
			edges += get_edges_general_alt(elem, prev = prev, prefix = '%s%d_' % (prefix, i))
	else:
		return [(prev, input_data)]
	return edges

def json_to_graph_alt(input_data):
	G = nx.Graph()
	nodes = get_nodes_general_alt(input_data)
	edges = get_edges_general_alt(input_data)
	G.add_nodes_from(nodes)
	print(edges)
	for edge in edges:
		try:
			G[edge[0]][edge[1]]['weight']+=1					
		except:
			G.add_edge(edge[0],edge[1],weight=1)
	return G

input_json = {
	'a' : {
		'b' : [{'c' : ['d1', 'd2'], 'e1': 'f1'}, 'c1', 'c2'], 'd' : 'e'
	},
	'e' : {
		'f' : 'g', 'h' : 'i', 'd': 'a'
	}

}

# input_json = {
# 	'a' : {
# 		'b' : 'c', 'd' : 'e'
# 	},
# 	'e' : {
# 		'f' : 'g', 'h' : 'i'
# 	}
# }

alt_input = [
      {
         "id": "X999_Y999",
         "from": {
            "name": "Tom Brady", "id": "X12"
         },';/'
         "message": "Looking forward to 2010!",
         "actions": [
            {
               "name": "Comment",
               "link": "http://www.facebook.com/X999/posts/Y999"
            },
            {
               "name": "Like",
               "link": "http://www.facebook.com/X999/posts/Y999"
            }
         ],
         "type": "status",
         "created_time": "2010-08-02T21:27:44+0000",
         "updated_time": "2010-08-02T21:27:44+0000"
      },
      {
         "id": "X998_Y998",
         "from": {
            "name": "Peyton Manning", "id": "X18"
         },
         "message": "Where's my contract?",
         "actions": [
            {
               "name": "Comment",
               "link": "http://www.facebook.com/X998/posts/Y998"
            },
            {
               "name": "Like",
               "link": "http://www.facebook.com/X998/posts/Y998"
            }
         ],
         "type": "status",
         "created_time": "2010-08-02T21:27:44+0000",
         "updated_time": "2010-08-02T21:27:44+0000"
      }
   ]

G = json_to_graph_alt(input_json)

def plotter(G):
	plt.figure(1,figsize=(8,8))
	nx.draw(G)
	plt.savefig("atlas.png",dpi=300)	

nx.draw_networkx(G, with_labels = True)
plt.show()

# if __name__ == "__main__":
# 	path_spec_file = sys.argv[1]
# 	with open(path_spec_file) as f:
# 		specifications = yaml.load(f)
# 	G = builder(specifications)
# 	pprint('writing graph')
# 	nx.write_gexf(G,specifications['output_file'])
