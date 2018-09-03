# BMA (Bounded-rational multi-agent systems)

A modular implementation of Blahut-Arimoto type algorithms for the feed-forward multi-agent systems in information-theoretic bounded rationality developed in [Gottwald, Braun. Systems of bounded rational agents with information-theoretic constraints](https://doi.org/).

A particular multi-agent architecture is indexed by a type and shape (as introduced in the paper). The type labels the general structure of the multi-agent system, i.e. the Bayesian network of random variables and their interdependencies, whereas the shape is a list of  each node's agent occupation number, i.e. a shape of `[1,2,4]` means that the first random variable consists of one agent, the second of two, and the third of four agents. The implementation is modular in the sense that a given multi-agent systems consists of a set number of nodes (= random variables), that are connected as specified in the file `BMA_sysconfig.py`. Currently, the algorithm is configured for all cases with `N<4`, but arbitrary types can be implemented by simply updating this file.

## Examples

* One-step rate distortion:
```python
MA = BMA.MultiAgentSystem(tp=(-1,),shape=[1])
MA.initialize(U,beta=[2.0])       
MA.iterate()
```

* Two-step serial case:
```python
MA = BMA.MultiAgentSystem(tp=(0,),shape=[1,1])
MA.initialize(U,beta=[4.0,3.0],M=[10])
MA.iterate()
```

* Three-step purely hierarchical case:
```python
MA = BMA.MultiAgentSystem(tp=(1,5),shape=[1,3,6])
MA.initialize(U,beta=[5.0]*10)
MA.iterate()
```

Note that here, the utility function `U` must be a numpy array of shape `(N,K)`, where `N` is the number of world states, and `K` the number of actions.


See also [BA_class](https://github.com/sgttwld/BA_class) and [Interactive Jupyter-Notebooks](https://github.com/sgttwld/blahut-arimoto) for two different implementations of the one- and two-step cases.

## Overview

### Useful information
```python
BMA.get_types()                     # print all currently implemented types  
BMA.get_shapes(tp=(0,3),n=10)       # print all possible shapes for n number of agents
BMA.get_M(tp=(0,3),M=20)            # print the required intermediate dimensions
BMA.draw_graphs(tps,fsize=(14,10))  # show graphs of the types in the list tps
```

### Usage
```python
import modules.BMA_system as BMA                  
# setting up the system by type and shape of the architecture:
MA = BMA.MultiAgentSystem(tp=(0,3),shape=[1,1,8])
# drawing a graph of the architecture:
MA.graph()                          
# set utility, betas (one for each agent), cardinality of intermediate variables:
MA.initialize(U,beta=[5.0]*10,M=[20])
# running the Blahut-Arimoto algorithm:
MA.iterate()
```

### Interesting quantities

#### `MultiAgentSystem` object
```python
MA.tp       # type of the system
MA.shape    # shape of the system
MA.DKL      # list of the DKLs (Kullback-Leibler divergences) of each step/node/RV
MA.EU       # expected utility of the system
MA.FE       # free energy of the system
MA.U        # utility function
MA.dims     # cardinalities [N,M1,M2,...,K] of the variables of the system
MA.M        # cardinalities [M2,M5,...] of intermediate non-selector nodes
MA.pagw     # numpy array containing the throughput policy p(a|w)
MA.pw       # Dist object containing the world state distribution p(w)
MA.joint    # Dist object containing the joint p(w,x1,x2,...,a)
MA.nd_lst   # list of the names of the nodes in the system
MA.nd       # list of the nodes (Node objects) in the system
MA.ag       # list of the agents (Agent objects) in the system
```

#### `Dist`(ribution) object (defined in `BMA_support.py`)
```python
dist = MA.joint   # example of a Dist object
dist.val          # numpy array that contains the values of the "distribution"
dist.r            # list of indices of MA.dims that correspond to np.shape(dist.val)
```

#### `Node` object  (defined in `BMA_node.py`)
```python
nd = MA.nd[0]     # first node in the MultiAgentSystem instance MA
nd.name           # name of the random variable corresponding to the node
nd.ag             # list of agents (Agent objects) in this node
nd.DKL            # (non-averaged) DKL of this node, e.g. DKL(p(x|w),p(x)) for each w
nd.post           # Dist object containing the posterior of the node
nd.prior          # Dist object containing the prior of the node
nd.marg           # Dist object containing the marginal of the node's posterior  
nd.p_in           # Dist object containing the input distribution, e.g. p(w|x)
```

#### `Agent` object  (defined in `BMA_agent.py`)
```python
ag = MA.ag[0]     # first agent in the MulitAgentSystem instance MA
ag.post           # numpy array containing the posterior of the agent
ag.prior          # numpy array containing the prior of the agent
ag.p_in           # numpy array containing the input distribution for the agent
ag.DKL            # DKL of the agent
```
