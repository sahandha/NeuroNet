#! python3
import networkx as nx
from NeuroNet.Neuron import *
import matplotlib.patches as patches
from tqdm import tnrange, tqdm_notebook, tqdm

class Brain:

    def __init__(self, neurons=[], dt = 0.1, tend=200):
        self._Neurons      = neurons
        self._NeuronDict   = {}
        self.AddNeuronToDict()
        self._SynapseCount = 0
        self._t            = 0
        self._dt           = dt
        self._Tend         = tend
        self._TLen         = int(tend/dt)
        self.InitializeNetwork()
        self.NeuronPrimer()
        self.ComputeSynapseProbability()
        self._AverageConnectivity=[]

    def NeuronPrimer(self):
        if len(self._Neurons)==0:
            n=Neuron(1,a=0.8,b=0.7,tau=12.5,I=0.5)
            self._Neurons.append(n)
            self._NeuronsDict[0]=n

    def AddNeuronToDict(self):
        for n in self._Neurons:
            self._NeuronDict[n._ID]=n

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
        for i in tnrange(self._TLen,desc='Time'):
            self.Update(i)
            #self.NetworkProperties()

    def Update(self,i):
        self._t += self._dt
        for n in self._Neurons:
            self.SynapticActivity(n)
            n.Update(i)

    def SynapticActivity(self,neuron):

        for n in self._Neurons:
            prob = self._SynapseProbability[(neuron,n)]
            if self.SynapseQ(prob) and (neuron._ID!=n._ID):
                if len(neuron._SynapsedNeurons)<20:
                    neuron.AddSynapse(n)
                    self._SynapseCount += 1
                    self.AddEdge(neuron,n)

    def InitializeNetwork(self):
        self._Network      = nx.DiGraph()
        self._EdgeLabels   = {}
        self._NodeLabels   = {}
        for n in self._Neurons:
            if n._ID == 0:
                self._Network.add_node(n,pos=(n._x,n._y),color='#db203c')
            else:
                self._Network.add_node(n,pos=(n._x,n._y),color='#1941d3')
            self._NodeLabels[n]=n._ID

    def AddEdge(self,n1,n2):
        edgeData = self._Network.get_edge_data(n1,n2,default=0)
        if edgeData==0:
            self._Network.add_edge(n1, n2, weight=0.2)
        else:
            self._Network.add_edge(n1, n2,weight=edgeData['weight']+0.2)

        self._EdgeLabels[(n1,n2)]='{:2.1f}'.format(self._t)

    def NetworkProperties(self):
        self._AverageConnectivity.append(nx.average_node_connectivity(self._Network))
        #self._Laplacian = nx.normalized_laplacian_matrix(self._Network.to_undirected())
        #self._Eigs      = np.linalg.eigvals(self._Laplacian.A)
        #print("Largest eigenvalue:", max(self._Eigs))
        #print("Smallest eigenvalue:", min(self._Eigs))
        #plt.hist(self._Eigs,bins=100) # histogram with 100 bins
        #plt.xlim(0,2)  # eigenvalues between 0 and 2
        #plt.show()

    def DrawNetwork(self, edgelabels=False):
        warnings.filterwarnings("ignore")
        fig1 = plt.figure()
        ax = fig1.add_subplot(111, aspect='equal')
        plt.xlim((0,80))
        plt.ylim((0,80))

        pos = nx.get_node_attributes(self._Network,'pos')
        weights = [self._Network[u][v]['weight'] for u,v in self._Network.edges()]
        nodecolors = list(nx.get_node_attributes(self._Network, 'color').values())

        nx.draw(self._Network, pos, node_size=110,width=weights)
        nx.draw_networkx_nodes(self._Network, pos, node_size=110,node_color=nodecolors)
        nx.draw_networkx_labels(self._Network, pos,  labels=self._NodeLabels,font_color=[1,1,1],font_family='Times',font_size=8)
        nx.draw_networkx_edges(self._Network, pos, edge_color='#68f2df', width=weights,arrows=True)
        if edgelabels:
            nx.draw_networkx_edge_labels(self._Network, pos, edge_labels=self._EdgeLabels, label_pos=0.5,font_size=6)

        # Background color to represent the experiment dish.
        ax.add_patch(patches.Rectangle((10, 0),60,10,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((70, 10),10,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((10, 70),60,10,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((0, 10),10,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((10, 10),60,60,facecolor=[0.1,0.1,0.6],alpha=0.2))

        plt.show()


if __name__ == "__main__":
    tend = 400
    dt   = 0.1

    numNeurons = 50
    displaynum = 10
    neurons = []

    for i in range(numNeurons):
        if i==0:
            I=1
        else:
            I=0
        n = Neuron(i,dt=dt,tend=tend,a=0.8,b=0.7,tau=12.5,I=I)
        n.SetFlow(n.FHNFlow)
        neurons.append(n)

    b = Brain(neurons=neurons,dt=dt,tend=tend)
    b.Simulate()
