import networkx as nx
from collections import defaultdict
# def add_edges(E1,E2):


def merge_graphs(G1,G2):
	G = nx.Graph()
	G.add_nodes_from(G1.nodes(data=True)+G2.nodes(data=True))
	B1 = G1.edges(data=True)
	C = [(elem[0],elem[1],elem[2]['attribute']) for elem in B1]
	# adding the edges
	for elem in G2.edges(data=True):
		if (elem[0],elem[1],elem[2]['attribute']) in C:
			idx = C.index((elem[0],elem[1],elem[2]['attribute']))
			B1[idx][2]['weight'] += elem[2]['weight']
		else:
			B1.append(elem)
	G.add_edges_from(B1)
	return

def merge_graphs1(G1,G2):
	G = nx.Graph()
	G.add_nodes_from(G1.nodes(data=True)+G2.nodes(data=True))
	# builds a default dict for the edges\
	C = defaultdict(int)
	# we should account for the order of the edges
	for elem in G1.edges(data=True):
		# small heuristic here to just make sure that there are no repeated edges
		d = sorted([elem[0],elem[1]])
		C[(d[0],d[1],elem[2]['attribute'])] += elem[2]['weight']
	for elem in G2.edges(data=True):
		d = sorted([elem[0],elem[1]])
		C[(elem[0],elem[1],elem[2]['attribute'])] += elem[2]['weight']
	G.add_edges_from([(elem[0],elem[1],{'attribute':elem[2], 'weight':C[elem]}) for elem in C])
	return G


G1_name = 'output_graph/output_graph_6.gexf'
G2_name = 'output_graph/output_graph_7.gexf'

G_name = 'output_graph/merged_graph.gexf'
G1 = nx.read_gexf(G1_name)
G2 = nx.read_gexf(G2_name)

G = merge_graphs1(G1,G2)
nx.write_gexf(G,G_name)