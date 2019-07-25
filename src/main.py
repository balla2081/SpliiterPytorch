import torch
import argparse
from splitter import SplitterTrainer
from utils import tab_printer, read_graph


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run splitter with node2vec")
    
    ## input and output files
    parser.add_argument('--input', nargs='?', default='graph/karate.elist',
                        help='Input graph path')
    
     parser.add_argument("--persona-graph", nargs = "?", default = "graph/karate-persona.elist",
                        help = "Persona network path.")
    
    parser.add_argument("--persona-mapping", nargs = "?", default = "mapping/karate.json",
                        help = "Persona mapping output path.")

    parser.add_argument('--output-base', nargs='?', default='emb/karate.emb',
                        help='Base Embeddings path')
    
    parser.add_argument('--output-persona', nargs='?', default='emb/karate_persona.pkl',
                        help='Persona Embeddings path')
    
    
    ## hyper-parameter of base_embedding
    
    parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

    parser.add_argument('--base_iter', default=1, type=int,
                      help='Number of epochs in base embedding')
    
    parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')
    
    ## hyper-parameter of splitter
    
    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Learning rate. Default is 0.01.")

    parser.add_argument("--lambd",
                        type = float,
                        default = 0.1,
	                help = "Regularization parameter. Default is 0.1.")
    
    parser.add_argument("--negative-samples",
                        type = int,
                        default = 5,
	                help = "Negative sample number. Default is 5.")
    
    
    
    ## computation configuration
    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                help = "Random seed for PyTorch. Default is 42.")
    
    parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	## parameters for input graph type
    parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
                           
    parser.set_defaults(directed=False)

    return parser.parse_args()






def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    Saving the persona mapping and the embedding.
    """
    args = parse_args()
    print(args)
    torch.manual_seed(args.seed)
    tab_printer(args)
    graph = read_graph(args)

    trainer = SplitterTrainer(graph, args)
    trainer.fit()
    trainer.save_embedding()
    trainer.save_persona_graph_mapping()

if __name__ == "__main__":
    main()
