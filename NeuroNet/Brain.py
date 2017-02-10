import networkx as nx
import random
from NeuroNet.Neuron import *
from NeuroNet.Synapse import *

class Brain:
    def __init__(self, neurons=[], dt = 0.1, tend=200):
        self._Neurons      = neurons
        self._Synapses     = {}
        self._SynapseCount = 0
        self._t            = 0
        self._dt           = dt
        self._Tend         = tend
        self._TLen         = int(tend/dt)
        self.SetSynpaseProbability()
        self.InitializeNetwork()
        if len(self._Neurons)==0:
            self._Neurons.append(Neuron(1,a=0.8,b=0.7,tau=12.5,I=0.5))


    def InitializeNetwork(self):
        self._Network      = nx.DiGraph()
        self._Network.add_nodes_from(self._Neurons)
        self._EdgeLabels   = {}
        self._NodeLabels   = {}
        for n in self._Neurons:
            self._NodeLabels[n]=n._ID


    def SetSynpaseProbability(self, chance=True, likelyhood=1000):
        self._prob = likelyhood*[False]+[chance]

    def Simulate(self):
        for id, s in self._Synapses.items():
            self.AddEdge(s._ParentNeuron,s._ChildNeuron)

        for i in range(self._TLen):
            self.Update(i)

    def Update(self,i):
        self._t += self._dt
        for n in self._Neurons:
            self.SynapticActivity(n)
            n.Update(i)

    def SynapticActivity(self,neuron):
        if random.choice(self._prob):
            dneuron = random.choice(self._Neurons)
            if neuron._ID != dneuron._ID:
                self._SynapseCount += 1
                s = Synapse(self._SynapseCount,neuron,dneuron)
                #print('A synapse just formed between Neuron {} and Neuron {} at time = {:2.2f}s.'.format(neuron._ID,dneuron._ID,self._t))
                neuron._Synapses.append(s)
                #dneuron._Synapses[self._SynapseCount]="In"
                self._Synapses[self._SynapseCount]=s
                self.AddEdge(neuron,dneuron)


    def AddEdge(self,n1,n2):
        self._Network.add_edge(n1, n2)
        self._EdgeLabels[(n1,n2)]='{:2.1f}'.format(self._t)

    def DrawNetwork(self):
        warnings.filterwarnings("ignore")
        pos = nx.circular_layout(self._Network)
        nx.draw(self._Network, pos)
        nx.draw_networkx_nodes(self._Network, pos, node_color='#8899FF')
        nx.draw_networkx_labels(self._Network, pos, labels=self._NodeLabels,font_color=[1,1,1],font_family='Times')
        nx.draw_networkx_edges(self._Network, pos, edge_color='#889933',arrows=True)
        nx.draw_networkx_edge_labels(self._Network, pos, edge_labels=self._EdgeLabels, label_pos=0.5)

        plt.show()
