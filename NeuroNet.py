#! python3

import warnings
import numpy as np
from GeneralModel import *

class Neuron(GeneralModel):
    def __init__(self, ID, synapses=[], tstart=0, tend=200, dt=0.1, **params):
        GeneralModel.__init__(self, Name="Neuron {}".format(ID), tstart=tstart, tend=tend, dt=dt,**params)
        self._ID       = ID
        self._V        = 0
        self._w        = 0
        self.Initialize([self._V,self._w])
        self._Synapses = synapses

    def Flow(self, t, x, params):
        I   = params["I"]
        a   = params["a"]
        b   = params["b"]
        tau = params["tau"]

        V, w = x[0], x[1]

        dV = V - V**3/3 - w + I
        dw = (V + a - b*w)/tau

        return np.array([dV, dw])

    def UpdateSynapses(self):
        for s in self.Synapses:
            s.ParentNeuron._w + s._ChildNeuron._w

    def Update(self,i):
        self._GM.UpdateRK(i)
        self._Time = self._GM._t
        self._V = self._GM._x[0]
        self._w = self._GM._x[1]

class Synapse:
    def __init__(self, ID, upstream, downstream):
        self._ID = ID
        self.SetParentNeuron(upstream)
        self.SetChildNeuron(downstream)

    def SetParentNeuron(self, upstream):
        self._ParentNeuron = upstream
    def SetChildNeuron(self, downstream):
        self._ChildNeuron = downstream

class Brain:
    def __init__(self, neurons=[], dt = 0.1, tend=200):
        self._Neurons = neurons
        self._dt      = dt
        self._Tend    = tend
        self._TLen    = int(tend/dt)

        if len(self._Neurons)==0:
            self._Neurons.append(Neuron(1,a=0.8,b=0.7,tau=12.5,I=0.5))

    def Simulate(self):
        for i in range(self._TLen):
            self.Update(i)

    def Update(self,i):
        for n in self._Neurons:
            n.UpdateRK(i)


if __name__ == '__main__':
    b = Brain()
    b.Simulate()
