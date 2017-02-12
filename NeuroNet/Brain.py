import networkx as nx
from NeuroNet.Neuron import *
from NeuroNet.Synapse import *
import matplotlib.patches as patches

class Brain:
    def __init__(self, neurons=[], dt = 0.1, tend=200):
        self._Neurons      = neurons
        self._Synapses     = {}
        self._SynapseCount = 0
        self._t            = 0
        self._dt           = dt
        self._Tend         = tend
        self._TLen         = int(tend/dt)
        self.InitializeNetwork()
        if len(self._Neurons)==0:
            self._Neurons.append(Neuron(1,a=0.8,b=0.7,tau=12.5,I=0.5))
        self.ComputeSynapseProbability()

    def Distance2(self, a, b):
        return sum((a - b)**2)

    def ComputeSynapseProbability(self):
        probabilityMatrix = {}
        for ii in range(len(self._Neurons)):
            n1 = self._Neurons[ii]
            for jj in range(ii,len(self._Neurons)):
                n2 = self._Neurons[jj]
                if ii == jj:
                    d = 0
                else:
                    d = self.Distance2(np.array([n1._x,n1._y]), np.array([n2._x,n2._y]))
                probabilityMatrix[(n1,n2)] = np.exp(-d/200)
                probabilityMatrix[(n2,n1)] = np.exp(-d/200)
        self._SynapseProbability = probabilityMatrix

    def SynapseQ(self,probability):
        return random.random() < probability

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

        for n in self._Neurons:
            prob = self._SynapseProbability[(neuron,n)]
            if self.SynapseQ(prob) and (neuron._ID!=n._ID):
                if len(neuron._Synapses)<11:
                    self._SynapseCount += 1
                    s = Synapse(self._SynapseCount,neuron,n)
                    neuron._Synapses.append(s)
                    self._Synapses[self._SynapseCount]=s
                    self.AddEdge(neuron,n)


    def InitializeNetwork(self):
        self._Network      = nx.DiGraph()
        self._EdgeLabels   = {}
        self._NodeLabels   = {}
        for n in self._Neurons:
            self._Network.add_node(n,pos=(n._x,n._y),size=10)
            self._NodeLabels[n]=n._ID

    def AddEdge(self,n1,n2):
        self._Network.add_edge(n1, n2)
        self._EdgeLabels[(n1,n2)]='{:2.1f}'.format(self._t)

    def DrawNetwork(self, edgelabels=False):
        warnings.filterwarnings("ignore")
        fig1 = plt.figure()
        ax = fig1.add_subplot(111, aspect='equal')
        plt.xlim((0,80))
        plt.ylim((0,80))
        pos = nx.get_node_attributes(self._Network,'pos')
        sizes = [1]*len(self._NodeLabels)
        nx.draw(self._Network, pos)
        nx.draw_networkx_nodes(self._Network, pos, node_color='#8899FF')
        nx.draw_networkx_labels(self._Network, pos, labels=self._NodeLabels,font_color=[1,1,1],font_family='Times')
        nx.draw_networkx_edges(self._Network, pos, edge_color='#889933',arrows=True)
        if edgelabels:
            nx.draw_networkx_edge_labels(self._Network, pos, edge_labels=self._EdgeLabels, label_pos=0.5)

        ax.add_patch(patches.Rectangle((10, 0),60,10,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((70, 10),10,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((10, 70),60,10,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((0, 10),10,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((10, 10),60,60,facecolor=[0.1,0.1,0.6],alpha=0.2))

        plt.show()
