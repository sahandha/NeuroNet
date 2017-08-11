# Individual based model of network of neurons.

<!-- TOC depthFrom:2 depthTo:3 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Overview](#overview)
	- [Serial code](#serial-code)
	- [Running Serially on Bluewaters](#running-serially-on-bluewaters)
	- [Running Parallel on Bluewaters](#running-parallel-on-bluewaters)
- [Agent-based model](#agent-based-model)
- [Vectorized Simulation](#vectorized-simulation)
	- [Sample ensemble firing patterns](#sample-ensemble-firing-patterns)
	- [Sample network connectivity matrix](#sample-network-connectivity-matrix)

<!-- /TOC -->

## Overview
We build a network of neurons that can form synapses over time. When a single neuron is excited, through the connections formed, all the other neurons are excited as well. See the file [demo.ipynb](https://github.com/sahandha/NeuroNet/blob/master/examples/demo.ipynb) in the examples folder for a sample simulation.

### Serial code

The master branch of this repository implements the simulation in serial for running on your personal computer.

### Running Serially on Bluewaters

For a serial code that can be run on Bluewaters see, the branch title [BW](https://github.com/sahandha/NeuroNet/tree/BW).

### Running Parallel on Bluewaters

For parallel code for running on Bluewaters, see the code [BW_MPI](https://github.com/sahandha/NeuroNet/tree/BW_MPI).

**Note** all parallel simulations are implemented only as vectorized simulations only. Essentially the Agent-based modeling portion of this repository is for demo purposes only.

## Agent-based model

In the initial phase of the development, we started out with Agent-based modeling. This is a good first start as it allows for flexibility and ease of use in specifying model parameters. But it is computationally expensive, and hence, not suitable for large simulations.

Neurons are self excited (due to noise). For some network configurations, we have synchronization, see [HERE](https://github.com/sahandha/NeuroNet/blob/master/NeuronModel/Demo-Copy2.ipynb). When the network is too weak, no synchronization happens, see [HERE](https://github.com/sahandha/NeuroNet/blob/master/NeuronModel/Demo-Copy3.ipynb).

## Vectorized Simulation

Simulating the system using Agent-based modeling approach is computationally expensive. So for large networks we use a vectorized form of the simulation. The code is maintained in [./NeuronModel](./NeuronModel). Please see some example notebooks therein.

Some preliminary results:

### Sample ensemble firing patterns

This is a network that is fully developed. The neurons are self-activated due to noise. We can see that almost immediately the neurons synchronize.

![Adjacency matrix](./Images/TimeFrequency.png)

### Sample network connectivity matrix

The two axes represent neurons. The neurons are sorted by their distance to the origin. The strong correlation observed indicates that the neurons that are closer to one another are more likely to form connections than those that are further apart.

![Adjacency matrix](./Images/AdjacencyMatrix.png)
