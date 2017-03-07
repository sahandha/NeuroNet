#! python3
import networkx as nx
from NeuroNet.Neuron import *
import matplotlib.patches as patches
from tqdm import tnrange, tqdm_notebook, tqdm
import itertools

class Brain:

    def __init__(self, neurons=[], dt = 0.1, tend=200, connectionscale=20):
        '''
            This class handles the overarching organization.
        '''
        self._Neurons       = neurons
        self._NeuronsDict   = {}
        self.AddNeuronToDict()
        self._SynapseCount = 0
        self._t            = 0
        self._dt           = dt
        self._Tend         = tend
        self._TLen         = round(tend/dt)
        self._ConnectionScale = connectionscale
        self._AverageConnectivity=[]
        self._SynapseCountHistory = []
        self.ConstructNetwork()
        self.NeuronPrimer()
        self.ComputeSynapseProbability()
        self._ActiveNeurons = []
        self._InactiveNeurons = []


    def CollectActiveNeurons(self):
        self._ActiveNeurons=[]
        for n in self._Neurons:
            if n._ActiveQ:
                self._ActiveNeurons.append(n)
            else:
                self._InactiveNeurons.append(n)

    def NeuronPrimer(self):
        if len(self._Neurons)==0:
            n=Neuron(1,a=0.8,b=0.7,tau=12.5,I=0.5)
            self._Neurons.append(n)
            self._NeuronsDict[0]=n
        self._NeuronCount = len(self._Neurons)

    def AddNeuronToDict(self):
        for n in self._Neurons:
            self._NeuronsDict[n._ID]=n

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
                n1._Distance[n2] = np.sqrt(d)
                n2._Distance[n1] = np.sqrt(d)
                probabilityMatrix[(n1,n2)] = np.exp(-d/self._ConnectionScale)
                probabilityMatrix[(n2,n1)] = np.exp(-d/self._ConnectionScale)
        self._SynapseProbability = probabilityMatrix

    #def SynapseQ(self,probability):
    #    return random.random() < probability

    def Simulate(self):
        for i in tnrange(self._TLen,desc='Tot Sim'): #tnrange only works with Jupyter
            self.Update(i)

    def Update(self,i):
        self._t += self._dt
        self._SynapseCountHistory.append(self._SynapseCount)
        for n in self._Neurons:
            #self.SynapticActivity(n)
            n.Update(i)
        for n in self._Neurons:
            n._Input = 0

    def SynapticActivity(self,neuron):
        randArray = np.random.random(self._NeuronCount)

        for idx, n in enumerate(self._Neurons):
            prob = self._SynapseProbability[(neuron,n)]
            if (randArray[idx] < prob) and (neuron._ID!=n._ID):
                if neuron._SynapseCount<neuron._SynapseLimit:
                    neuron.AddSynapse(n)
                    self._SynapseCount += 0.5/self._NeuronCount
                    self.AddEdge(neuron,n)

    def DevelopSynapseNetwork(self):
        #[self.SynapticActivity(n) for n in self._Neurons]
        for n in self._Neurons:
            self.SynapticActivity(n)

    def ConstructNetwork(self):
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
