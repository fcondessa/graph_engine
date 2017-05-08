from __future__ import division
import networkx as nx

'''timeseries = [(start, end, [])]'''
class DynamicGraph(nx.Graph):
    def __init__(self, data = None, edges_timeseries = dict(), **attr):
        super(DynamicGraph, self).__init__(data)
        for edge in self.edges_iter():
            (u, v) = edge
            if edge in edges_timeseries:
                self[u][v]['timeseries'] = edges_timeseries[edge]
            else:
                if ('timeseries' not in self[u][v]):
                    weight = (self[u][v]['weight'] 
                        if 'weight' in self[u][v] else 1)
                    self[u][v]['timeseries'] = [(None, None, [weight])]
        self.graph.update(attr)
    
    def get_weight(self, u, v, t):
        timeseries = self[u][v]['timeseries']
        weight = 0
        for entry in timeseries:
            (start, end, data) = entry
            if ((start == None or t >= start) and (end == None or t < end)):
                if (start == None or end == None):
                    assert(len(data) == 1)
                    weight += data[0]
                else:
                    interval = (end - start)/len(data)
                    index = int((t - start)//interval)
                    weight += data[index]
        return weight

    def at(self, t):
        for edge in self.edges_iter():
            u, v = edge
            self[u][v]['weight'] = self.get_weight(u, v, t)
        return self

def merge_fn(G, edge):
    (u, v, timeseries) = (edge[0], edge[1], edge[2]['timeseries'])
    if G.has_edge(u, v):
        G[u][v]['timeseries'] += timeseries
    else:
        G.add_edge(u, v, edge[2])