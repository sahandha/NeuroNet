# Individual based model of network of neurons.

We build a network of neurons that can form synapses over time. When a single neuron is excited, through the connections formed, all the other neurons are excited as well. See the file [demo.ipynb](https://github.com/sahandha/NeuroNet/blob/master/examples/demo.ipynb) in the examples folder for a sample simulation.

Neurons are self excited (due to noise). For some network configurations, we have synchronization, see [HERE](https://github.com/sahandha/NeuroNet/blob/master/NeuronModel/Demo-Copy2.ipynb). When the network is too weak, no synchronization happens, see [HERE](https://github.com/sahandha/NeuroNet/blob/master/NeuronModel/Demo-Copy3.ipynb).

## Agent-based model

In the initial phase of the development, we started out with Agent-based modeling. This is a good first start as it allows for flexibility and ease of use in specifying model parameters. But it is computationally expensive, and hence, not suitable for large simulations.

--------------------------------------------------------------------------------

## Vectorized Simulation

Simulating the system using Agent-based modeling approach is computationally expensive. So for large networks we use a vectorized form of the simulation. The code is maintained in <./NeuronModel>. Please see some example notebooks therein.

Some preliminary results:

### Sample ensemble firing patterns

![Adjacency matrix](./Images/TimeFrequency.png)

### Sample network connectivity matrix

![Adjacency matrix](./Images/AdjacencyMatrix.png)
