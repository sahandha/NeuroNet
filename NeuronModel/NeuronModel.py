#!/usr/bin/env python
#######! python3
import numpy as np
import itertools as it
from mpi4py import MPI
import copy

class NeuronModel():
    Comm  = MPI.COMM_WORLD
    def __init__(self, N=10, t0=0, tend=100, dt=0.1, connectionscale=50, synapselimit=1000, synapsestrengthlimit=50, networkdevel=10, **params):
        self._dt                   = dt
        self._tstart               = t0
        self._tend                 = tend
        self._t                    = t0
        self._NetworkDevel         = networkdevel
        self._Time                 = np.arange(self._tstart,self._tend,self._dt)
        self._X                    = np.array([])
        self._dX                   = np.array([])
        self._Xp                   = np.array([])
        self._dXp                  = np.array([])
        self._NumberOfNeurons      = N
        self._NoiseMean            = 0
        self._NoiseSTD             = 0.015
        self._ConnectionScale      = connectionscale
        self._SynapseCount         = np.zeros(self._NumberOfNeurons)
        self._SynapseLimit         = synapselimit
        self._SynapseStrengthLimit = synapsestrengthlimit
        self._CellType             = np.random.choice([-1,1],size=N,p=[1/5,4/5])
        self._NetworkDevelTime     = networkdevel
        self._NeuronPosition       = []
        self._Distance             = {}
        #self._SynapseProbability   = {}
        self._DelayIndx            = np.zeros(self._NumberOfNeurons,dtype=np.int)
        #self._SynapseWeight        = {key: 0 for key in it.product(range(self._NumberOfNeurons),repeat=2)}
        #self._SynapseQ             = {key: False for key in it.product(range(self._NumberOfNeurons),repeat=2)}
        self._Params               = params
        self.PlaceNeurons()
        self.ComputeDelayIndex()
        #self.ComputeDistances()
        #self.DevelopNetwork(networkdevel)
        self.Initialize()

    def SetStorage(self,s):
        self._Storage = s

    def PlaceNeurons(self):
        for n in range(self._NumberOfNeurons):
            np.random.seed(n)
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

    def ComputeDistances(self):
        #for ii in range(self._NumberOfNeurons):
        #    for jj in range(ii,self._NumberOfNeurons):
        #        if ii == jj:
        #            d                                 = 0
        #            self._Distance[(ii,jj)]           = 0
        #        else:
        #            d = np.sqrt(self.Distance2(self._NeuronPosition[ii], self._NeuronPosition[jj]))
        #            self._Distance[(ii,jj)]           = d
        #            self._Distance[(jj,ii)]           = d

    def GetDelayIndex(self,n1,n2):
        if n1==n2:
            d = 0
        else:
            d = np.sqrt(self.Distance2(self._NeuronPosition[n1], self._NeuronPosition[n2]))
        return d

    def ComputeDelayIndex(self):
        for i in range(self._NumberOfNeurons):
            self._DelayIndx = np.array([int(2*self.GetDelayIndex(i,n)/self._dt) for n in range(self._NumberOfNeurons)])

    def Distance2(self, a, b):
        return sum((a - b)**2)

    def DevelopNetwork(self,n):
        for key,value in self._Distance.items():
            if key[0]==key[1]:
                self._SynapseWeight[key] = 0
            else:
                self._SynapseWeight[key]=min(int(n*np.exp(-value/self._ConnectionScale)),self._SynapseLimit)

    def GetWeight(self, n1, n2):
        if n1==n2:
            w = 0
        else:
            d = np.sqrt(self.Distance2(self._NeuronPosition[n1], self._NeuronPosition[n2]))
            w = min(int(self._NetworkDevelTime*np.exp(-d/self._ConnectionScale)), self._SynapseLimit)

        if self._Storage._WriteNetwork:
            self._Storage.WriteNetworkItem(n1,n2,w,NeuronModel.Comm.rank)

        return w

    def Initialize(self):
        self._V    = np.random.normal(-40,1,size=self._NumberOfNeurons)
        self._N    = np.zeros(self._NumberOfNeurons)
        self._X    = np.concatenate((self._V, self._N))
        self._dV   = np.zeros_like(self._V)
        self._dN   = np.zeros_like(self._N)
        self._dX   = np.zeros_like(self._X)
        self._Time = np.arange(self._tstart,self._tend,self._dt)
        self._dim  = len(self._X)
        self._VV   = np.zeros((max(self._DelayIndx),self._NumberOfNeurons))
        self.SetParameters()
        self._Input = np.zeros(self._NumberOfNeurons)

        self.SplitData()

    def SplitData(self):
        cs = NeuronModel.Comm.size
        s  = int(self._NumberOfNeurons/cs)
        r = NeuronModel.Comm.rank
        self._V   = self._X[:self._NumberOfNeurons]
        self._N   = self._X[self._NumberOfNeurons:]
        self._Vp  = copy.copy(self._V[s*r:s*(r+1)])
        self._Np  = copy.copy(self._N[s*r:s*(r+1)])
        self._dVp = copy.copy(self._dV[s*r:s*(r+1)])
        self._dNp = copy.copy(self._dN[s*r:s*(r+1)])
        self._Xp  = np.concatenate((self._Vp, self._Np))
        self._dXp = np.concatenate((self._dVp, self._dNp))
        self._Ip  = copy.copy(self._I[s*r:s*(r+1)])
        self._Inputp = copy.copy(self._Input[s*r:s*(r+1)])

    def SetParameters(self):
        params = self._Params
        cs = NeuronModel.Comm.size
        s  = int(self._NumberOfNeurons/cs)
        self._I   = np.random.normal(params["I"  ],params["I_v"  ],size=s)
        self._C   = np.random.normal(params["C"  ],params["C_v"  ],size=s)
        self._gCa = np.random.normal(params["gCa"],params["gCa_v"],size=s)
        self._VCa = np.random.normal(params["VCa"],params["VCa_v"],size=s)
        self._gK  = np.random.normal(params["gK" ],params["gK_v" ],size=s)
        self._VK  = np.random.normal(params["VK" ],params["VK_v" ],size=s)
        self._gL  = np.random.normal(params["gL" ],params["gL_v" ],size=s)
        self._VL  = np.random.normal(params["VL" ],params["VL_v" ],size=s)
        self._phi = np.random.normal(params["phi"],params["phi_v"],size=s)
        self._V1  = np.random.normal(params["V1" ],params["V1_v" ],size=s)
        self._V2  = np.random.normal(params["V2" ],params["V2_v" ],size=s)
        self._V3  = np.random.normal(params["V3" ],params["V3_v" ],size=s)
        self._V4  = np.random.normal(params["V4" ],params["V4_v" ],size=s)

    def UpdateSynapses(self,indx):
        cs = NeuronModel.Comm.size
        s  = int(self._NumberOfNeurons/cs)
        r = NeuronModel.Comm.rank

        if indx != 0 and self._Storage._WriteNetwork:
            self._Storage.CloseNetworkFile(r)

        self._Inputp = np.zeros_like(self._Vp)
        for i in range(s):
            input = self._VV[-self._DelayIndx,np.arange(self._NumberOfNeurons)]
            weights = np.array([self.GetWeight(n,r*s+i) for n in range(self._NumberOfNeurons)])
            self._Inputp[i] = sum(1/self._SynapseLimit*weights*self._CellType*1/(1+np.exp(-input)))

    def MLFlow(self, t, x):

        V = self._Vp
        N = self._Np

        Mss = 0.5*(1+np.tanh((V-self._V1)/self._V2))
        Nss = 0.5*(1+np.tanh((V-self._V3)/self._V4))
        Tau = 1/(self._phi*np.cosh((V-self._V3)/self._V4/2))

        self._dVp = (self._I - self._gL*(V-self._VL) - self._gCa*Mss*(V-self._VCa) - self._gK*N*(V-self._VK))/self._C + self._Inputp
        self._dNp = (Nss - N)/Tau

        return np.concatenate((self._dVp, self._dNp))

    def UpdateRK(self,ii):
        cs = NeuronModel.Comm.size
        s  = int(self._NumberOfNeurons/cs)
        r = NeuronModel.Comm.rank
        k1 = self.MLFlow(self._t, self._Xp)
        k2 = self.MLFlow(self._t+self._dt/2, self._Xp+k1*self._dt/2)
        k3 = self.MLFlow(self._t+self._dt/2, self._Xp+k2*self._dt/2)
        k4 = self.MLFlow(self._t+self._dt  , self._Xp+k2*self._dt  )
        self._Xp = self._Xp + (k1 + 2*k2 + 2*k3 + k4)*self._dt/6
        self._t = self._t + self._dt

        self._Vp = self._Xp[:s]
        self._Np = self._Xp[s:]

    def UpdateEuler(self,ii):
        self._Xp = self._Xp + self._dt*self.MLFlow(self._t, self._Xp);
        self._t  = self._t + self._dt
        self._Vp = self._Xp[:s]
        self._Np = self._Xp[s:]

    def StoreTimeSeriesData(self, i):
        self._VV = np.concatenate((self._VV[1:,:], self._V.reshape(1,len(self._V))))

    def AddNoise(self,indx):
        s  = int(self._NumberOfNeurons/NeuronModel.Comm.size)
        self._X[self._NumberOfNeurons:] += np.random.normal(self._NoiseMean, self._NoiseSTD, self._NumberOfNeurons)
        self._Np += np.random.normal(self._NoiseMean, self._NoiseSTD, s)

    def Simulate(self, source='Jupyter'):
        self.Initialize()

        self._dXp = np.array(self.MLFlow(self._t, self._Xp))


        if source=='Jupyter':
            for ii in range(len(self._Time)):
                self.WriteData()
                self.StoreTimeSeriesData(ii)
                self.AddNoise(ii)
                self.UpdateSynapses(ii)
                self.MPICOMM()
                self.UpdateRK(ii);
                if NeuronModel.Comm.rank == 0:
                    print("step ",ii, " of ", len(self._Time))
        else:
            for ii in range(len(self._Time)):
                self.WriteData()
                self.StoreTimeSeriesData(ii)
                self.AddNoise(ii)
                self.UpdateSynapses(ii)
                self.UpdateRK(ii);
                self.MPICOMM()


    def MPICOMM(self):
        NeuronModel.Comm.Barrier()
        NeuronModel.Comm.Allgather( [self._Vp, MPI.DOUBLE], [self._V, MPI.DOUBLE] )
        NeuronModel.Comm.Allgather( [self._Np, MPI.DOUBLE], [self._N, MPI.DOUBLE] )
        NeuronModel.Comm.Allgather( [self._dVp, MPI.DOUBLE], [self._dV, MPI.DOUBLE] )
        NeuronModel.Comm.Allgather( [self._dNp, MPI.DOUBLE], [self._dN, MPI.DOUBLE] )
        #NeuronModel.Comm.Gather( [self._Inputp, MPI.DOUBLE], [self._Input, MPI.DOUBLE] )
        self._X = np.concatenate((self._V,self._N))
        self._dX = np.concatenate((self._dV,self._dN))


    def WriteData(self):
        if NeuronModel.Comm.rank == 0:
            self._Storage.WriteLine()


if __name__=='__main__':
    comm = NeuronModel.Comm
    print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
    comm.Barrier()   # wait for everybody to synchronize _here_
