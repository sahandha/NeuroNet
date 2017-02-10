#! python3
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
        self._Models["FittzHuge-Nagamo"] = self.FHNFlow
        self._Models["Simple-Ionic"]     = self.SIFlow

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
        self._V = self._x[0]
        self._w = self._x[1]

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
