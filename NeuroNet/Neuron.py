#! python3
from GeneralModel import GeneralModel
import numpy as np
class Neuron(GeneralModel):
    def __init__(self, ID, synapses=[], tstart=0, tend=200, dt=0.1, **params):
        '''
            The Neuron object
        '''
        GeneralModel.__init__(self, Name="Neuron {}".format(ID),tstart=tstart, tend=tend, dt=dt,**params)
        self._ID    = ID
        self._CellType = 1 # 1 for excitatory, -1 for inhibatory
        self._eps   = params["eps"] # for scaling time.
        self._V     = 0
        self._w     = 0
        self._s     = 1
        self._Input = 0
        self.Initialize([self._V,self._w])
        self._Synapses = synapses
        self._II       = np.zeros_like(self._Time) #important it comes afeter initialization.
        self._Models = {}  # remove SIR, Vander-pol and other models.
        self._Models["FittzHuge-Nagamo"] = self.FHNFlow
        self._Models["Simple-Ionic"]     = self.SIFlow
        self._Models["Morris-Lecar"]     = self.MLFlow
        self._Distance = {}
        self.PlaceNeuron()
        self._SynapsedNeuronsDict = {}
        self._SynapseCount = 0
        self._ActiveQ = False
        self._SynapseLimit=1000
        self._NoiseMean = 0
        self._NoiseSTD = 0.01
        self._Noise = np.random.normal(self._NoiseMean, self._NoiseSTD,len(self._Time))
        self._SynapticStrength = 0.01#/self._SynapseLimit#random.uniform(0.1,0.11)

    def SetNoise(self,m, v):
        self._NoiseMean = m
        self._NoiseSTD = v
        self._Noise = np.random.normal(self._NoiseMean, self._NoiseSTD,len(self._Time))

    def SetSynapseLimit(self, lim):
        self._SynapseLimit = lim

    def AddSynapse(self,n):
        try:
            self._SynapsedNeuronsDict[n] += 1
        except:
            self._SynapsedNeuronsDict[n] = 1

        self._SynapseCount += 1

    def AvailableModels(self):
        print(list(self._Models.keys()))

    def PlaceNeuron(self):
        x = 80*np.random.random() #range between 0, 80
        y = 80*np.random.random() #range between 0, 80
        if x<=10 and y<=10:
            self._x = x + (50*np.random.random()+10) #range between 10, 60
            self._y = y + (50*np.random.random()+10) #range between 10, 60
        elif x>=70 and y<=10:
            self._x = x - (50*np.random.random()+10) #range between 10, 60
            self._y = y + (50*np.random.random()+10) #range between 10, 60
        elif x<=10 and y>=70:
            self._x = x + (50*np.random.random()+10) #range between 10, 60
            self._y = y - (50*np.random.random()+10) #range between 10, 60
        elif x>=70 and y>=70:
            self._x = x - (50*np.random.random()+10) #range between 10, 60
            self._y = y - (50*np.random.random()+10) #range between 10, 60
        else:
            self._x = x
            self._y = y

        self._Cell = (int(np.floor(self._x/10)),int(np.floor(self._y/10)))

    def SetInput(self, i):
        self._Input = i

    def GetW(self):
        return self._w

    def GetV(self):
        return self._V

    def GetS(self):
        return self._s

    def SetV(self, v):
        self._V = v

    def SetW(self, w):
        self._w = w

    def SetS(self, s):
        self._s = s

    def UpdateSynapses(self):
        self._Delay = 0
        for n, s in self._SynapsedNeuronsDict.items():
            delayTime = 2*n._eps*self._Distance[n]*2 #TODO: Units need to be sorted out
            delayIdx  = int((n._t-delayTime)/n._dt)
            if delayIdx > 0:
                potential = n._XX[delayIdx,1]
                if potential > 0:
                    input = potential
                else:
                    input = 0
            else:
                input = 0

            if n._ID != self._ID:
                self._Input += n._CellType*s*n._SynapticStrength*1/(1+np.exp(-input))


    def Update(self,i):
        self.UpdateSynapses()
        self.UpdateEuler(i)
        self._V = self._X[0]
        self._w = self._X[1]
        if self._V > 1:
            self._ActiveQ = True
        self.AddNoise(i)
        self.StoreInputHistory(i)

    def AddNoise(self,i):
        self._X[1] += self._Noise[i]

    def StoreInputHistory(self,i):
        self._II[i] = self._Input #+ self._params["I"]

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

        if self._t > 200*self._eps:
            I=0

        V, w = x[0], x[1]

        dV = (V - V**3/3 - w + I + self._Input)/self._eps
        dw = ((V + a - b*w)/tau)/self._eps
        return np.array([dV, dw])

    def MLFlow1(self, t, x, params):
        I   = params["I"]

        C   = self._params["C"  ]
        gCa = self._params["gCa"]
        VCa = self._params["VCa"]
        gK  = self._params["gK" ]
        VK  = self._params["VK" ]
        gL  = self._params["gL" ]
        VL  = self._params["VL" ]

        V, w = x[0], x[1]
        dV = (I - gCa*self.m_inf(V)*(V-VCa) - gK*w*(V-VK)-gL*(V-VL))/C
        dw = self.alpha(V)*(1-w)-self.beta(V)*w
        return np.array([dV, dw])

    def m_inf(self, v):
        V1 = self._params["V1"]
        V3 = self._params["V3"]
        return 0.5*(1+np.tanh((v-V1)/V3))

    def alpha(self, v):
        phi = self._params["phi"]
        V3  = self._params["V3"]
        V4  = self._params["V4"]
        return 0.5*phi*np.cosh((v-V3)/(2*V4))*(1+np.tanh((v-V3)/V4))

    def beta(self, v):
        phi = self._params["phi"]
        V3  = self._params["V3"]
        V4  = self._params["V4"]
        return 0.5*phi*np.cosh((v-V3)/(2*V4))*(1-np.tanh((v-V3)/V4))

    def MLFlow(self, t, x, params):
        I   = self._params["I"]
        C   = self._params["C"  ]
        gCa = self._params["gCa"]
        VCa = self._params["VCa"]
        gK  = self._params["gK" ]
        VK  = self._params["VK" ]
        gL  = self._params["gL" ]
        VL  = self._params["VL" ]
        phi = self._params["phi"]
        V1  = self._params["V1"]
        V2  = self._params["V2"]
        V3  = self._params["V3"]
        V4  = self._params["V4"]

        if self._t > 250:
            I = 0

        V, N = x[0], x[1]
        Mss = 0.5*(1+np.tanh((V-V1)/V2))
        Nss = 0.5*(1+np.tanh((V-V3)/V4))
        Tau = 1/(phi*np.cosh((V-V3)/(2*V4)))
        n   = len(self._SynapsedNeuronsDict)
        dV = ( I - gL*(V-VL) - gCa*Mss*(V-VCa) - gK*N*(V-VK))/C + self._Input
        dN = (Nss - N)/Tau
        return np.array([dV, dN])


    def Hev(self, x):
        return 0.5 * (np.sign(x) + 1)
