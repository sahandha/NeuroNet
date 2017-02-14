#! python3
import networkx as nx
from NeuroNet.Neuron import *
import matplotlib.patches as patches
from tqdm import tnrange, tqdm_notebook, tqdm
import itertools

class Brain:

    def __init__(self, neurons=[], dt = 0.1, tend=200, connectionscale=2,synapserate=10):
        self._Neurons      = neurons
        self._NeuronDict   = {}
        self.AddNeuronToDict()
        self._SynapseCount = 0
        self._t            = 0
        self._dt           = dt
        self._Tend         = tend
        self._TLen         = int(tend/dt)
        self._ConnectionScale = connectionscale
        self._SynapseRate     = synapserate
        self._AverageConnectivity=[]
        self._SynapseLimit = 100
        self._SynapseCountHistory = []
        self.InitializeNetwork()
        self.NeuronPrimer()
        self.ComputeSynapseProbability()
        self._NeuronPairs = list(itertools.permutations(self._Neurons,2))



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
                probabilityMatrix[(n1,n2)] = np.exp(-d/self._ConnectionScale)
                probabilityMatrix[(n2,n1)] = np.exp(-d/self._ConnectionScale)
        self._SynapseProbability = probabilityMatrix

    def SynapseQ(self,probability):
        return random.random() < probability

    def Simulate(self):
        for i in tnrange(self._TLen,desc='Time'): #tnrange only works with Jupyter
            self.Update(i)
            #self.NetworkProperties()

    def UpdateSpecial(self,i):
        self._t += self._dt
        self._SynapseCountHistory.append(self._SynapseCount)

        cn = self._NeuronPairs[0][0]
        cn.Update(i)
        for np in self._NeuronPairs:
            self.SynapticActivitySpecial(np)
            if np[0]!= cn:
                cn.Update(i)
                cn = np[0]

    def Update(self,i):
        self._t += self._dt
        self._SynapseCountHistory.append(self._SynapseCount)
        for n in self._Neurons:
            if self._t%self._SynapseRate < self._dt/2:
                self.SynapticActivity(n)
            n.Update(i)

    def SynapticActivitySpecial(self,neuronpair):
        prob = self._SynapseProbability[neuronpair]
        n1, n2 = neuronpair[0],neuronpair[1]
        if self.SynapseQ(prob) and len(n1._SynapsedNeurons)<self._SynapseLimit:
            n1.AddSynapse(n2)
            self._SynapseCount += 1
            self.AddEdge(n1,n2)

    def SynapticActivity(self,neuron):

        for n in self._Neurons:
            prob = self._SynapseProbability[(neuron,n)]
            if self.SynapseQ(prob) and (neuron._ID!=n._ID):
                if len(neuron._SynapsedNeurons)<self._SynapseLimit:
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
            self._Network.add_edge(n1, n2, weight=1)
        else:
            self._Network.add_edge(n1, n2,weight=edgeData['weight']+1)

        self._EdgeLabels[(n1,n2)]='{:2.1f}'.format(self._t)

    def PlotConnectivityProperties(self):

        #self._AverageConnectivity.append(nx.average_node_connectivity(self._Network))

        plt.subplot(131)
        self._DegreeDistribution = sorted(nx.degree(self._Network).values(),reverse=True)
        p=plt.loglog(self._DegreeDistribution,'-',marker='o')
        plt.setp(p,color='darkblue')

        plt.title("Degree rank plot")
        plt.ylabel("degree")
        plt.xlabel("rank")

        plt.subplot(132)
        plt.hist([len(n._SynapsedNeurons) for n in self._Neurons], 20,facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
        plt.title('Distribution of number of synapses formed')
        plt.xlabel('Number of synapses')
        plt.ylabel('Count')

        plt.subplot(133)
        weights = [self._Network.get_edge_data(n1,n2,default=0)['weight'] for n1,n2 in self._Network.edges()]
        plt.hist(weights, 20, facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
        plt.title('Distribution of edge weights')
        plt.xlabel('Edge weight')
        plt.ylabel('Count')

        plt.show()

    def DrawNetwork(self, edgelabels=False):
        warnings.filterwarnings("ignore")
        fig1 = plt.figure()
        ax = fig1.add_subplot(111, aspect='equal')
        plt.xlim((0,80))
        plt.ylim((0,80))


        for n in self._Neurons:
            if n._ActiveQ and n._ID != 0:
                nx.set_node_attributes(self._Network, 'color', {n:'#918b21'})

        pos = nx.get_node_attributes(self._Network,'pos')
        weights = [2/self._SynapseLimit*self._Network[u][v]['weight'] for u,v in self._Network.edges()]
        nodecolors = list(nx.get_node_attributes(self._Network, 'color').values())

        nx.draw(self._Network, pos, node_size=110, width=weights)
        nx.draw_networkx_nodes(self._Network, pos, node_size=110,node_color=nodecolors)
        nx.draw_networkx_labels(self._Network, pos,  labels=self._NodeLabels,font_color=[1,1,1],font_family='Times',font_size=8)
        nx.draw_networkx_edges(self._Network, pos, edge_color='#2c8c7d', width=weights,arrows=True)
        if edgelabels:
            nx.draw_networkx_edge_labels(self._Network, pos, edge_labels=self._EdgeLabels, label_pos=0.5,font_size=6)

        # Background color to represent the experiment dish.
        ax.add_patch(patches.Rectangle((10, 0),60,10,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((70, 10),10,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((10, 70),60,10,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((0, 10),10,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((10, 10),60,60,facecolor=[0.1,0.1,0.6],alpha=0.2))

        plt.show()

    def ComputeEigenValues(self,matrix="Laplacian"):
        if matrix=="Laplacian":
            L = nx.directed_laplacian_matrix(self._Network)
            eigs = np.linalg.eigvals(L)
        elif matrix=="Adjacency":
            eigs = sorted(nx.adjacency_spectrum(self._Network))
        return eigs

    def PlotEigenValues(self):
        plt.figure

        plt.subplot(121)
        eigs = self.ComputeEigenValues(matrix="Adjacency")
        reals = [np.real(n) for n in eigs]
        imag  = [np.imag(n) for n in eigs]
        plt.plot(reals,imag,'o')
        plt.grid(True)
        plt.title('Eigenvalues of Adjacency Matrix')
        plt.xlabel('real part')
        plt.ylabel('imaginary part')

        plt.subplot(122)
        eigs = self.ComputeEigenValues(matrix="Laplacian")
        reals = [np.real(n) for n in eigs]
        imag  = [np.imag(n) for n in eigs]
        plt.plot(reals,imag,'o')
        plt.grid(True)
        plt.title('Eigenvalues of Laplacian Matrix')
        plt.xlabel('real part')


        plt.show()

    def PlotDegreeDistribution(self):
        idegree = list(self._Network.in_degree().values())
        odegree = list(self._Network.out_degree().values())

        try:
            plt.subplot(131)
            plt.hist(idegree, max(idegree), facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
            plt.title('in-degree')
            plt.subplot(132)
            plt.hist(odegree, max(odegree), facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
            plt.title('out-degree')
            plt.subplot(133)
            plt.hist(self._DegreeDistribution, max(self._DegreeDistribution), facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
            plt.title('total degree distribution')
            plt.show()
        except:
            print("No connections at all")

    def PlotSynapseRank(self):
        p = plt.plot(self._Neurons[0]._Time,self._SynapseCountHistory)
        plt.setp(p, 'Color', [0.6,0.4,0.5], 'linewidth', 3)
        plt.grid(True)
        plt.ylabel("Count")
        plt.xlabel("Time")
        plt.title('Synapse Count Increase Over Time')
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
