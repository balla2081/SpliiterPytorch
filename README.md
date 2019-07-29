# SplitterPytorch
A PyTorch implementation of "Splitter: Learning Node Representations that Capture Multiple Social Contexts" (WWW 2019).

> Splitter: Learning Node Representations that Capture Multiple Social Contexts.
> Alessandro Epasto and Bryan Perozzi.
> WWW, 2019.
> [[Paper]](http://epasto.org/papers/www2019splitter.pdf)

This code is based on https://github.com/benedekrozemberczki/Splitter for the splitter codes, and node2vec https://github.com/eliorc/node2vec. I reorganize the codes and fix some bugs for my research. So,this codes works pretty well on emperical dataset, and very easy to use. The original Tensorflow implementation is available [[here]](https://github.com/google-research/google-research/tree/master/graph_embedding/persona). But there is only codes for generating persona, not embedding codes...

### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             0.4.1
gensim            3.6.0
```
### Datasets - inputs
The code takes the **edge list** of the graph in a csv file. You can easily make the edgelist file with networkx function [nx.write_edgelist](https://networkx.github.io/documentation/networkx1.10/reference/generated/networkx.readwrite.edgelist.write_edgelist.html

### Outputs
There are 4 outputs on Splitter

1. Persona graph
  Persona graph is a result graph of ego-splitting. File format is edgelist, which is same to format of inputs
  
2. Persona mapping
  Persona mapping is a dict that connnect orginal node and splitted persona nodes. Bascially, relation between two entities is 1 to M relations. File format is json.
  
3. Base embedding
  Result embedding of Node2vec on orignal graph, it used as base embedding of splitter. File format is word2vec fomrat of gensim. You can read a result with gensim.models.Keyedvectors.load_word2vec_format()
  
4. Persona embedding
  Result embedding of Spitter on persona graph. This embedding is final results of this resposiotry. File format is pickle(.pkl).

### Options
The training of a Splitter embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options
   
```
  --input [INPUT]       Input graph path
  --persona-graph [PERSONA_GRAPH]
                        Persona network path.
  --persona-mapping [PERSONA_MAPPING]
                        Persona mapping path.
  --emb_base [EMB_BASE]
                        Base Embeddings path
  --emb_persona [EMB_PERSONA]
                        Persona Embeddings path
```
#### Model options
```
    --dimensions DIMENSIONS
                        Number of dimensions. Default is 128.
  --walk-length WALK_LENGTH
                        Length of walk per source. Default is 80.
  --num-walks NUM_WALKS
                        Number of walks per source. Default is 10.
  --window-size WINDOW_SIZE
                        Context size for optimization. Default is 10.
  --base_iter BASE_ITER
                        Number of epochs in base embedding
  --p P                 Return hyperparameter for random-walker. Default is 1.
  --q Q                 Inout hyperparameter for random-walker. Default is 1.
  --learning-rate LEARNING_RATE
                        Learning rate. Default is 0.01.
  --lambd LAMBD         Regularization parameter. Default is 0.1.
  --negative-samples NEGATIVE_SAMPLES
                        Negative sample number. Default is 5.
  --seed SEED           Random seed for PyTorch. Default is 42.
  --workers WORKERS     Number of parallel workers. Default is 8.
  --weighted            Boolean specifying (un)weighted. Default is
                        unweighted.
  --unweighted
  --directed            Graph is (un)directed. Default is undirected.
  --undirected
```

### Examples
The following commands learn an embedding and save it with the persona map. Training a model on the default dataset.
```
python src/main.py
```

Training a Splitter model with 32 dimensions.
```
python src/main.py --dimensions 32
```
Increasing the number of walks and the walk length.
```
python src/main.py --number-of-walks 20 --walk-length 80
```
