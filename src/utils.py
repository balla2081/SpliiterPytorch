import pandas as pd
import networkx as nx
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def read_graph(args):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=str, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=str, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G
