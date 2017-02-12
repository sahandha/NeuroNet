#! python3
import random
from GeneralModel import *
class Neuron(GeneralModel):
    def __init__(self, ID, synapses=[], tstart=0, tend=200, dt=0.1, **params):
        GeneralModel.__init__(self, Name="Neuron {}".format(ID),tstart=tstart, tend=tend, dt=dt,**params)
        self._ID    = ID
        self._V     = 0
        self._w     = 0
        self._Input = params["I"]
        self.Initialize([self._V,self._w])
        self._Synapses = synapses
        self._II       = np.zeros_like(self._Time) #important it comes afeter initialization.
        self._Models = {}  # remove SIR, Vander-pol and other models. 
        self._Models["FittzHuge-Nagamo"] = self.FHNFlow
        self._Models["Simple-Ionic"]     = self.SIFlow
        self.PlaceNeuron()

    def AvailableModels(self):
        print(list(self._Models.keys()))

    def PlaceNeuron(self):
        x = random.randrange(0,80)
        y = random.randrange(0,80)
        if x<=10 and y<=10:
            self._x = x + random.randrange(10,60)
            self._y = y + random.randrange(10,60)
        elif x>=70 and y<=10:
            self._x = x - random.randrange(10,60)
            self._y = y + random.randrange(10,60)
        elif x<=10 and y>=70:
            self._x = x + random.randrange(10,60)
            self._y = y - random.randrange(10,60)
        elif x>=70 and y>=70:
            self._x = x - random.randrange(10,60)
            self._y = y - random.randrange(10,60)
        else:
            self._x = x
            self._y = y

        self._Cell = (int(np.floor(self._x/10)),int(np.floor(self._y/10)))

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

    def UpdateSynapses(self):
        for s in self._Synapses:
            s.Update()

    def Update(self,i):
        self.StoreInputHistory(i)
        self.UpdateSynapses()
        self.UpdateRK(i)
        self._V = self._X[0]
        self._w = self._X[1]

    def StoreInputHistory(self,i):
        self._II[i] = self._Input



    #Coupled Inhibitory Oscillation Model

    def s_ji(self,V,**params):
        theta_syn = params["theta_syn"]
        k_syn     = params["k_syn"]
        return 1/(1+np.exp(-(V-theta_syn)/k_syn))
    def m_infty(self,V):
        return 1/(1+np.exp(-(V+65)/7.8))
    def h_infty(self,V):
        return 1/(1+np.exp((V+81)/11))
    def tau_h(self,V):
        return self.h_infty(V)*np.exp((V+162.3)/17.8)

    def SIFlow(self, t, x, params):
        g_pir     = params["g_pir"]
        V_pir     = params["V_pir"]
        g_L       = params["g_L"]
        V_L       = params["V_L"]
        g_syn     = params["g_syn"]
        V_syn     = params["V_syn"]
        phi       = params["phi"]
        tau_0     = params["tau_0"]
        theta_syn = params["theta_syn"]
        k_syn     = params["k_syn"]

        V, h = x[0], x[1]
        dV = -g_pir*self.m_infty(V)**3 * h *(V - V_pir) - g_L*(V-V_L) - g_syn*self.s_ji(self._Input,theta_syn=theta_syn,k_syn=k_syn)*(V-V_syn)
        dh = (phi*(self.h_infty(V)-h))/self.tau_h(V)

        return np.array([dV, dh])


    def FHNFlow(self, t, x, params):
        I   = params["I"]
        a   = params["a"]
        b   = params["b"]
        tau = params["tau"]

        V, w = x[0], x[1]

        dV = V - V**3/3 - w + self._Input
        dw = (V + a - b*w)/tau
        return np.array([dV, dw])
