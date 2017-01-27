from __future__ import print_function
import warnings
import numpy as np

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

    def Flow(self, t):
        dV = self._V - self._V**3/3 - self._w - self.Input
        dw = 1/self._tau*(self._V + self._a - self._b*self._w)
        return np.array([dV, dw])

    def Flow(self, t):

        if self._State == "Action":
            if self._V >= min:
                dV = 5*t
                dw =
            if self._V >= self._Threshold:
                dV = 10*t
            if self._V >= max:
                dv = -10*t
                self._state = "Relaxation"

        elif self._State == "Relaxation":
            if self._V < min:
                dV = 5*t
                dw =
                if self._V >= self._Threshold:
                    dV = 10*t
                if self._V >= max:
                    dv = -10*t
                    self._state = "Relaxation"



        return np.array([dV, dw])

    def Update_V(self,t):
        return 5*t

    def Update(self):
        self._V = self.Update_V(self._Time)
        print("The value of voltage is: {}".format(self._V))

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
    def __init__(neurons=[], dt = 0.01):

        for n in neurons:
            n.Update()
            
