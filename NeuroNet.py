#! python3

import warnings
import numpy as np
from GeneralModel import *

class Neuron:
    def __init__(self, ID, synapses=[], time=0, a=0.7, b=0.8, tau=0.8, I=0):
        self._ID       = ID
        self._Time     = time
        self._Input    = I
        self._V        = 0
        self._w        = 0
        self._a        = a
        self._b        = b
        self._tau      = tau
        self._Synapses = synapses

        self._GM = GeneralModel(Name="Neuron_{}".format(self._ID), tstart=self._Time, tend=50, dt=0.01)
        self._GM.Initialize([1,0])
        self._GM.SetFlow(self.Flow_FHN)

    def Flow_FHN(self, t, x, params):
        V, w = x[0], x[1]
        dV = V - V**3/3 - w - self._Input
        dw = 1/self._tau*(V + self._a - self._b*w)
        return np.array([dV, dw])

    def Flow(self, t, x, params):

        if self._State == "Action":
            if self._V >= min:
                dV = 5*t
                dw = 0
            if self._V >= self._Threshold:
                dV = 10*t
            if self._V >= max:
                dv = -10*t
                self._state = "Relaxation"

        elif self._State == "Relaxation":
            if self._V < min:
                dV = 5*t
                dw = 0
                if self._V >= self._Threshold:
                    dV = 10*t
                if self._V >= max:
                    dv = -10*t
                    self._state = "Relaxation"



        return np.array([dV, dw])

    def UpdateSynapses(self):
        for s in self.Synapses:
            s.ParentNeuron._w + s._ChildNeuron._w

    def Update(self):
        self._GM.UpdateRK()
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
    def __init__(self, neurons=[], dt = 0.01, tend=50):
        self._Neurons = neurons
        self._dt      = dt
        self._Tend    = tend

        if len(self._Neurons)==0:
            self._Neurons.append(Neuron(1))

    def Simulate(self):
        for i in np.arange(0,10,0.1):
            for n in self._Neurons:
                n.Update()
            print("Neuron {}".format(n._ID))
            print("Voltage: {}".format(n._V))
            print("Current: {}".format(n._w))
            print("========================")


if __name__ == '__main__':
    b = Brain()
    b.Simulate()
