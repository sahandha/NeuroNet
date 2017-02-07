#! python3

import warnings
import numpy as np
from GeneralModel import *
import random
import networkx as nx

class Neuron(GeneralModel):
    def __init__(self, ID, synapses=[], tstart=0, tend=200, dt=0.1, **params):
        GeneralModel.__init__(self, Name="Neuron {}".format(ID), tstart=tstart, tend=tend, dt=dt,**params)
        self._ID    = ID
        self._V     = 0
        self._w     = 0
        self._Input = params["I"]
        self.Initialize([self._V,self._w])
        self._Synapses = synapses
        self._II       = np.zeros_like(self._Time) #important it comes afeter initialization.

    def setInput(self, i):
        self._Input = i

    def getW(self):
        return self._w

    def getV(self):
        return self._V

    def setV(self, v):
        self._V = v

    def setW(self, w):
        self._w = w



    def Flow(self, t, x, params):
        I   = params["I"]
        a   = params["a"]
        b   = params["b"]
        tau = params["tau"]

        #wsign = np.sign(x[1])
        #absw  = abs(x[1])
        #V, w = x[0], wsign*min(absw,2)

        V, w = x[0], x[1]

        #insign = np.sign(self._Input)
        #absin  = abs(self._Input)
        #self._Input = insign*min(absin,2)

        dV = V - V**3/3 - w + self._Input
        dw = (V + a - b*w)/tau

        return np.array([dV, dw])

    def UpdateSynapses(self):
        for s in self._Synapses:
            s.Update()

    def Update(self,i):
        self.StoreInputHistory(i)
        self.UpdateSynapses()
        self.UpdateRK(i)
        self._V = self._x[0]
        self._w = self._x[1]

    def StoreInputHistory(self,i):
        self._II[i] = self._Input

class Synapse:
    def __init__(self, ID, upstream, downstream):
        self._ID   = ID
        self._SynapticStrength = 0.1
        self.SetParentNeuron(upstream)
        self.SetChildNeuron(downstream)

    def SetParentNeuron(self, upstream):
        self._ParentNeuron = upstream
    def SetChildNeuron(self, downstream):
        self._ChildNeuron = downstream

    def Update(self):
        self._ChildNeuron.setInput(self._SynapticStrength*self._ParentNeuron.getW())


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

if __name__ == '__main__':
    b = Brain()
    b.Simulate()
    b._Neurons[0].PlotState(states={0:"V",1:"w"},legend=["Voltage","Current"])
