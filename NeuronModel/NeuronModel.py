#! python3
import numpy as np
import itertools as it

class NeuronModel():

    def __init__(self, N=10, t0=0, tend=100, dt=0.1, connectionscale=50, synapselimit=1000, synapsestrengthlimit=50, networkdeveltime=10, **params):
        self._dt              = dt
        self._tstart          = t0
        self._tend            = tend
        self._t               = t0
        self._NetworkDevelTime= networkdeveltime
        self._Time            = np.arange(self._tstart,self._tend,self._dt)
        self._X               = np.array([])
        self._dX              = np.array([])
        self._NumberOfNeurons = N
        self._NoiseMean       = 0
        self._NoiseSTD        = 0.03
        self._ConnectionScale = connectionscale
        self._SynapseCount    = np.zeros(self._NumberOfNeurons)
        self._SynapseLimit    = synapselimit
        self._SynapseStrengthLimit = synapsestrengthlimit
        self._CellType        = np.random.choice([-1,1],size=N,p=[1/5,4/5])
        self._NeuronPosition  = []
        self._Distance        = {}
        self._SynapseProbability = {}
        self._SynapseWeight   = {key: 0 for key in it.product(range(self._NumberOfNeurons),repeat=2)}
        self._SynapseQ        = {key: False for key in it.product(range(self._NumberOfNeurons),repeat=2)}
        self._Params          = params
        self.PlaceNeurons()
        self.Initialize()
        self.ComputeDistances() #This will result in self_DelayIndx and self._Distance
        # self.DevelopNetwork() #This will result in self._EdgeWieghts = {(n1,n2):w,...}

    def SetStorage(self,s):
        self._Storage = s

    def PlaceNeurons(self):
        for n in range(self._NumberOfNeurons):
            x = 80*np.random.random() #range between 0, 80
            y = 80*np.random.random() #range between 0, 80
            if x<=10 and y<=10:
                x = x + (50*np.random.random()+10) #range between 10, 60
                y = y + (50*np.random.random()+10) #range between 10, 60
            elif x>=70 and y<=10:
                x = x - (50*np.random.random()+10) #range between 10, 60
                y = y + (50*np.random.random()+10) #range between 10, 60
            elif x<=10 and y>=70:
                x = x + (50*np.random.random()+10) #range between 10, 60
                y = y - (50*np.random.random()+10) #range between 10, 60
            elif x>=70 and y>=70:
                x = x - (50*np.random.random()+10) #range between 10, 60
                y = y - (50*np.random.random()+10) #range between 10, 60
            else:
                x = x
                y = y
            self._NeuronPosition.append(np.array([x,y]))
            #self._Cell = (int(np.floor(self._x/10)),int(np.floor(self._y/10)))

    def ComputeDistances(self):
        for ii in range(self._NumberOfNeurons):
            for jj in range(ii,self._NumberOfNeurons):
                if ii == jj:
                    d                                 = 0
                    self._Distance[(ii,jj)]           = 0
                    self._SynapseProbability[(ii,jj)] = 0
                else:
                    d = np.sqrt(self.Distance2(self._NeuronPosition[ii], self._NeuronPosition[jj]))
                    self._Distance[(ii,jj)]           = d
                    self._Distance[(jj,ii)]           = d
                    self._SynapseProbability[(ii,jj)] = np.exp(-d/self._ConnectionScale)
                    self._SynapseProbability[(jj,ii)] = np.exp(-d/self._ConnectionScale)
        for i in range(self._NumberOfNeurons):
            self._DelayIndx = np.array([int(2*self._Distance[(i,n)]/self._dt) for n in range(self._NumberOfNeurons)])

    def Distance2(self, a, b):
        return sum((a - b)**2)

    def DevelopNetwork(self,source='Jupyter'):
        for i in range(self._NumberOfNeurons):
            wd = np.array([self.GetWeight(n,i) for n in range(self._NumberOfNeurons)])
            self._Weights[i] = wd[:,0]
            self._Delays[i]  = wd[:,1]
            self._Storage.WriteNetworkGroup(i,wd[:,0])
        '''
        x = 0;
        self._NetworkDevel = n
        if source=='Jupyter':
            for t in range(n):
                for key in self._SynapseWeight.keys():
                    if self._SynapseCount[key[0]] < self._SynapseLimit and self._SynapseCount[key[1]] < self._SynapseLimit:
                        if np.random.random()<self._SynapseProbability[key]:
                            self._SynapseQ[key]        = True
                            self._SynapseCount[key[0]]+=1
                            self._SynapseCount[key[1]]+=1

                    if self._SynapseQ[key] and self._SynapseWeight[key] < self._SynapseStrengthLimit:
                        self._SynapseWeight[key]+=1


        else:
            for t in range(n): #trange(n):
                for key in self._SynapseWeight.keys():
                    if np.random.random()<self._SynapseProbability[key]:
                        if self._SynapseWeight[key] < self._SynapseStrengthLimit:
                            self._SynapseWeight[key]+=1 #self._SynapseLimit/n
                        if self._SynapseCount[key[0]] < self._SynapseLimit and self._SynapseCount[key[1]] < self._SynapseLimit:
                            self._SynapseCount[key[0]]+=1
                            self._SynapseCount[key[1]]+=1
        '''

    def GetWeight(self, n1=1, n2=2):
        if n1==n2:
            w = 0
            delay = 1
        else:
            d = np.sqrt(self.Distance2(self._NeuronPosition[n1], self._NeuronPosition[n2]))
            w = min(int(self._NetworkDevelTime*np.exp(-d/self._ConnectionScale)), self._SynapseLimit)
            delay = int(2*d/self._dt)
        return (w,delay)

    def Initialize(self):
        self._X    = np.concatenate((np.random.normal(-40,1,size=self._NumberOfNeurons), np.zeros(self._NumberOfNeurons)))
        self._dX   = np.zeros_like(self._X)
        self._Time = np.arange(self._tstart,self._tend,self._dt)
        self._dim  = len(self._X)
        self._VV   = np.zeros((len(self._Time),self._NumberOfNeurons))
        self._dVV  = np.zeros((len(self._Time),self._NumberOfNeurons))
        self._NN   = np.zeros_like(self._VV)
        self._dNN  = np.zeros_like(self._dVV)
        self._Weights = {}
        self._Delays = {}
        self.SetParameters()
        self._DelayIndx = np.zeros(self._NumberOfNeurons,dtype=np.int8)
        self._Input = np.zeros(self._NumberOfNeurons)

    def SetParameters(self):
        params = self._Params
        self._I   = np.random.normal(params["I"  ],params["I_v"  ],size=self._NumberOfNeurons)
        self._C   = np.random.normal(params["C"  ],params["C_v"  ],size=self._NumberOfNeurons)
        self._gCa = np.random.normal(params["gCa"],params["gCa_v"],size=self._NumberOfNeurons)
        self._VCa = np.random.normal(params["VCa"],params["VCa_v"],size=self._NumberOfNeurons)
        self._gK  = np.random.normal(params["gK" ],params["gK_v" ],size=self._NumberOfNeurons)
        self._VK  = np.random.normal(params["VK" ],params["VK_v" ],size=self._NumberOfNeurons)
        self._gL  = np.random.normal(params["gL" ],params["gL_v" ],size=self._NumberOfNeurons)
        self._VL  = np.random.normal(params["VL" ],params["VL_v" ],size=self._NumberOfNeurons)
        self._phi = np.random.normal(params["phi"],params["phi_v"],size=self._NumberOfNeurons)
        self._V1  = np.random.normal(params["V1" ],params["V1_v" ],size=self._NumberOfNeurons)
        self._V2  = np.random.normal(params["V2" ],params["V2_v" ],size=self._NumberOfNeurons)
        self._V3  = np.random.normal(params["V3" ],params["V3_v" ],size=self._NumberOfNeurons)
        self._V4  = np.random.normal(params["V4" ],params["V4_v" ],size=self._NumberOfNeurons)

    def updateSynapses(self,indx):
        EffectiveIndx = indx + self._DelayIndx;

        self._Input = np.zeros(self._NumberOfNeurons)
        for i in range(self._NumberOfNeurons):
            input = self._VV[EffectiveIndx,np.arange(self._NumberOfNeurons)]
            weights = np.array([self._SynapseWeight[(n,i)] for n in range(self._NumberOfNeurons)])
            self._Input[i] = sum(1/self._SynapseLimit*weights*self._CellType*1/(1+np.exp(-input)))
        self._Input[EffectiveIndx<0] = 0

    def MLFlow(self, t, x):

        V = self._X[:self._NumberOfNeurons]
        N = self._X[self._NumberOfNeurons:]

        Mss = 0.5*(1+np.tanh((V-self._V1)/self._V2))
        Nss = 0.5*(1+np.tanh((V-self._V3)/self._V4))
        Tau = 1/(self._phi*np.cosh((V-self._V3)/self._V4/2))

        dV = (self._I - self._gL*(V-self._VL) - self._gCa*Mss*(V-self._VCa) - self._gK*N*(V-self._VK))/self._C + self._Input
        dN = (Nss - N)/Tau

        return np.concatenate((dV, dN))

    def UpdateRK(self,ii):
        #self._XX[ii,:] = self._X;
        #self._dXX[ii,:] = self._dX;
        k1 = self.MLFlow(self._t, self._X)
        k2 = self.MLFlow(self._t+self._dt/2, self._X+k1*self._dt/2)
        k3 = self.MLFlow(self._t+self._dt/2, self._X+k2*self._dt/2)
        k4 = self.MLFlow(self._t+self._dt  , self._X+k2*self._dt  )
        self._X = self._X + (k1 + 2*k2 + 2*k3 + k4)*self._dt/6
        self._t = self._t + self._dt

    def UpdateEuler(self,ii):
        #self._XX[ii,:] = self._X;
        #self._dXX[ii,:] = self._dX;

        self._dX = self.Flow(self._t, self._X, self._params);
        self._X  = self._X + self._dt*self._dX;
        self._t  = self._t + self._dt

    def StoreTimeSeriesData(self, i):
        self._VV[i,:] = self._X[:self._NumberOfNeurons]
        self._NN[i,:] = self._X[self._NumberOfNeurons:]
        self._dVV[i,:] = self._dX[:self._NumberOfNeurons]
        self._dNN[i,:] = self._dX[self._NumberOfNeurons:]

    def AddNoise(self,indx):
        self._X[self._NumberOfNeurons:] += np.random.normal(self._NoiseMean, self._NoiseSTD, self._NumberOfNeurons)

    def Simulate(self, source='Jupyter'):
        self._dX = np.array(self.MLFlow(self._t, self._X))
        self.Initialize()

        if source=='Jupyter':
            for ii in range(len(self._Time)):
                self.WriteData()
                self.StoreTimeSeriesData(ii)
                self.updateSynapses(ii)
                self.AddNoise(ii)
                self.UpdateRK(ii);
        else:
            for ii in range(len(self._Time)):
                self.WriteData()
                self.StoreTimeSeriesData(ii)
                self.updateSynapses(ii)
                self.AddNoise(ii)
                self.UpdateRK(ii);

    def WriteData(self):
        self._Storage.WriteLine()
