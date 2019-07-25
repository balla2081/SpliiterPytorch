import json
import torch
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from walkers import Node2Vec
from ego_splitting import EgoNetSplitter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class Splitter(torch.nn.Module):
    """
    An implementation of "Splitter: Learning Node Representations that Capture Multiple Social Contexts" (WWW 2019).
    Paper: http://epasto.org/papers/www2019splitter.pdf
    """
    def __init__(self, dimensions, lambd, base_node_count, node_count, device):
        """
        Splitter set up.
        :param dimensions: Dimension of embedding vectors
        :param lambd: Parameter that determine how much perosnas spread from originl embeddding
        :param base_node_count: Number of nodes in the source graph.
        :param node_count: Number of nodes in the persona graph.
        :param device: Deveice which pytorch use 
        """
        super(Splitter, self).__init__()

        self.dimensions = dimensions
        self.lambd = lambd
        self.base_node_count = base_node_count
        self.node_count = node_count
        self.device = device

    def create_weights(self):
        """
        Creating weights for embedding.
        """
        self.base_node_embedding = torch.nn.Embedding(self.base_node_count, self.dimensions, padding_idx = 0)
        self.node_embedding = torch.nn.Embedding(self.node_count, self.dimensions, padding_idx = 0)

    def initialize_weights(self, base_node_embedding, mapping, str2idx):
        """
        Using the base embedding and the persona mapping for initializing the embedding matrices.
        :param base_node_embedding: Node embedding of the source graph.
        :param mapping: Mapping of personas to nodes.
        """
        persona_embedding = np.array([base_node_embedding[str2idx[original_node]] for node, original_node in mapping.items()])
        self.node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(persona_embedding))
        self.base_node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(base_node_embedding), requires_grad=False)

    def calculate_main_loss(self, node_f, feature_f, targets):
        node_f = torch.nn.functional.normalize(node_f, p=2, dim=1).to(self.device)
        feature_f = torch.nn.functional.normalize(feature_f, p=2, dim=1).to(self.device)
        scores = torch.sum(node_f*feature_f, dim=1).to(self.device)
        scores = torch.sigmoid(scores).to(self.device)
        main_loss = targets*torch.log(scores) + (1-targets)*torch.log(1-scores).to(self.device)
        main_loss = -torch.mean(main_loss).to(self.device)
        
        return main_loss

    def calculate_regularization(self, source_f, original_f):
        source_f = torch.nn.functional.normalize(source_f, p=2, dim=1).to(self.device)
        original_f = torch.nn.functional.normalize(original_f, p=2, dim=1).to(self.device)
        scores = torch.sum(source_f*original_f,dim=1).to(self.device)
        scores = torch.sigmoid(scores).to(self.device)
        regularization_loss = -torch.mean(torch.log(scores)).to(self.device)
        
        return regularization_loss

    

    def forward(self, node_f, feature_f, targets, source_f, original_f): 
        main_loss = self.calculate_main_loss(node_f, feature_f, targets)
        regularization_loss = self.calculate_regularization(source_f, original_f)
        loss = main_loss + self.lambd * regularization_loss
        
        return loss.to(self.device)
         
class SplitterTrainer(object):
    """
    Class for training a Splitter.
    """
    def __init__(self, graph, 
                        directed=False,
                        num_walks=10,
                        walk_length=80,
                        p=1,
                        q=1,
                        dimensions=128,
                        window_size=10,
                        base_iter=1,
                        learning_rate=0.01,
                        lambd = 0.1,
                        negative_samples=5,
                        workers=1):
        """
        :param graph: NetworkX graph object.
        :param args: Arguments object.
        """
        self.graph = graph
        self.directed = directed

        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.dimensions = dimensions 
        self.window_size = window_size
        self.base_iter = base_iter
        self.workers = workers

        self.learning_rate = learning_rate
        self.lambd = lambd
        self.negative_samples = negative_samples 

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    def create_negative_sample_pool(self):
        self.downsampled_degrees = {node: int(1+self.egonet_splitter.persona_graph.degree(node)**0.75) for node in self.egonet_splitter.persona_graph.nodes()}
        self.negative_samples_pool = [k for k,v in self.downsampled_degrees.items() for i in range(v)]
                  
    def base_model_fit(self):
        """
        Fitting Node2Vec base model.
        """
        self.base_walker = Node2Vec(self.graph,
                                            directed=self.directed,
                                            num_walks=self.num_walks,
                                            walk_length=self.walk_length,
                                            p=self.p,
                                            q=self.q,
                                            dimensions=self.dimensions,
                                            window_size=self.window_size,
                                            base_iter=self.base_iter,
                                            workers=self.workers)
        logging.info("Doing base random walks.")
        self.base_walker.simulate_walks()
        logging.info("Learning the base model.")
        self.base_node_embedding = self.base_walker.learn_embedding()
        del self.base_walker.walks

    def create_split(self):
        """
        Creating an EgoNetSplitter.
        """
        self.egonet_splitter = EgoNetSplitter(self.graph)
        self.persona_walker = Node2Vec(self.egonet_splitter.persona_graph,
                                            directed=self.directed,
                                            num_walks=self.num_walks,
                                            walk_length=self.walk_length,
                                            p=self.p,
                                            q=self.q)
        logging.info("Doing persona random walks.")
        self.persona_walker.simulate_walks()
        self.create_negative_sample_pool()

    def setup_model(self):
        """
        Creating a model and doing a transfer to GPU.
        """
        base_node_count = self.graph.number_of_nodes()
        persona_node_count = self.egonet_splitter.persona_graph.number_of_nodes()
        self.model = Splitter(self.dimensions, self.lambd, base_node_count, persona_node_count, self.device)
        self.model.create_weights()
        self.model.initialize_weights(self.base_node_embedding, self.egonet_splitter.personality_map, self.base_walker.str2idx) 

    def reset_node_sets(self):
        """
        Resetting the node sets.
        """
        self.pure_sources = []
        self.personas = []
        self.sources = []
        self.contexts = []
        self.targets = []       

    def sample_negative_nodes(self):
        """
        Sampling noise nodes for context.
        """
        negative_nodes = np.random.choice(self.negative_samples_pool, self.negative_samples)
        return list(negative_nodes)

    def create_batch(self, source_node, context_node):
        """
        Augmenting a batch of data.
        :param source_node: A source node.
        :param context_node: A target to predict.
        """
        self.pure_sources += [source_node]
        self.personas += [self.base_walker.str2idx[self.egonet_splitter.personality_map[source_node]]]
        self.sources  += [source_node]*(self.negative_samples + 1)
        self.contexts +=[context_node] + self.sample_negative_nodes()
        self.targets += [1.0] + [0.0]*self.negative_samples     

    def transfer_batch(self):
        """
        Transfering the batch to GPU.
        """
        self.node_f = self.model.node_embedding(torch.LongTensor(self.sources)).to(self.device)
        self.feature_f = self.model.node_embedding(torch.LongTensor(self.contexts)).to(self.device)
        self.targets = torch.FloatTensor(self.targets).to(self.device)
        self.source_f = self.model.node_embedding(torch.LongTensor(self.pure_sources)).to(self.device)
        self.original_f = self.model.base_node_embedding(torch.LongTensor(self.personas)).to(self.device)

    def optimize(self):
        """
        Doing a weight update.
        """
        loss = self.model(self.node_f, self.feature_f, self.targets, self.source_f, self.original_f)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.reset_node_sets()
        return loss.item()  

    def fit(self):
        """
        Fitting a model.
        """
        self.reset_node_sets()
        self.base_model_fit()
        self.create_split()
        self.setup_model()
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.zero_grad()
        logging.info("Learning the joint model.")
        np.random.shuffle(self.persona_walker.walks)
        self.steps = 0
        self.losses = 0
        self.walk_steps = trange(len(self.persona_walker.walks), desc="Loss")
        for step in self.walk_steps:                
            walk = self.persona_walker.walks[step]
            for i in range(self.walk_length-self.window_size):
                for j in range(1,self.window_size+1):
                    source_node = walk[i]
                    context_node = walk[i+j]
                    self.create_batch(source_node, context_node)
            for i in range(self.window_size,self.walk_length):
                for j in range(1,self.window_size+1):
                    source_node = walk[i]
                    context_node = walk[i-j]
                    self.create_batch(source_node, context_node)
            
            if (step + 1) % self.num_walks == 0:
                self.transfer_batch()
                self.losses = self.losses + self.optimize()
                self.steps = self.steps + 1
                average_loss = self.losses / self.steps
                self.walk_steps.set_description("Splitter (Loss=%g)" % round(average_loss,4))
                
            if (step + 1) % 1000 ==0:
                self.steps = 0
                self.losses = 0

    # save functions 

    def save_base_embedding(self, file_name):
        self.base_walker.save_embedding(file_name)      
        
    def save_persona_embedding(self, file_name):
        """
        Saving the node embedding.
        """
        logging.info("Saving the model.")
        nodes = [node for node in self.egonet_splitter.persona_graph.nodes()]
        nodes.sort()
        nodes = torch.LongTensor(nodes)
        self.embedding = self.model.node_embedding(nodes).detach().numpy()
        return_data = {str(node.item()): embedding for node, embedding in zip(nodes, self.embedding)}
        pd.to_pickle(return_data, file_name)
                
    def save_persona_graph_mapping(self, file_name):
        """
        Saving the persona map.
        """
        with open(file_name, "w") as f:
            json.dump(self.egonet_splitter.personality_map, f)

    def save_persona_graph(self, file_name):
        nx.write_edgelist(self.egonet_splitter.persona_graph, file_name)

