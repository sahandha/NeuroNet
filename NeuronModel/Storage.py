import json
import os
import numpy as np
class Storage:
    def __init__(self, DataFolder, NeuronsPerFile, brain=None, NumberOfFiles=None, NumberOfNeurons=None, ParameterFileName=None):
        self._Brain          = brain
        self._DataFolder     = DataFolder
        if not os.path.exists(self._DataFolder):
            os.makedirs(self._DataFolder)
        self._ParameterFileName = ParameterFileName
        self.GetParams(NumberOfNeurons,NumberOfFiles,NeuronsPerFile)
        self._FileNames={}
        self._FullData=[]
        self.AssembleFileNames()


    @classmethod
    def FromFile(cls, name):
        with open(name) as data_file:
            data = json.load(data_file)
        NeuronsPerFile  = data["NeuronsPerFile"]
        DataFolder      = data["DataFolder"]
        NumberOfFiles   = data["NumberOfFiles"]
        NumberOfNeurons = data["NumberOfNeurons"]
        ConnectionScale = data["ConnectionScale"]
        NetworkDevel    = data["NetworkDevel"]

        return cls(DataFolder, NeuronsPerFile=NeuronsPerFile, NumberOfFiles=NumberOfFiles, NumberOfNeurons=NumberOfNeurons, ParameterFileName=name)

    def GetParams(self,NumberOfNeurons,NumberOfFiles,NeuronsPerFile):

        if self._Brain == None:
            with open(self._ParameterFileName) as data_file:
                data = json.load(data_file)
            self._tend = data["tend"]
            self._dt   = data["dt"]
            self._ConnectionScale   = data["ConnectionScale"]
            self._SynapseLimit   = data["SynapseLimit"]
            self._NeuronsPerFile = NeuronsPerFile
            self._NumberOfNeurons = NumberOfNeurons
            self._NumberOfFiles   = NumberOfFiles
            self._Parameters      = data
        else:
            self._NeuronsPerFile  = NeuronsPerFile
            self._NumberOfNeurons = self._Brain._NumberOfNeurons
            self._NumberOfFiles   = max(1,round(0.5+self._Brain._NumberOfNeurons/NeuronsPerFile))
            self._Parameters      = self._Brain._Params

    def AssembleFileNames(self):
        for i in range(self._NumberOfFiles):
            name = self._DataFolder + "/data" + str(i).zfill(4) + ".dat"
            self._FileNames[i] = name

    def ConstructLine(self,fileID):
        l = "{:.3f}".format(self._Brain._t) + ', '
        for i in range(fileID*self._NeuronsPerFile,min((fileID+1)*self._NeuronsPerFile,self._Brain._NumberOfNeurons)):
                l += "{:.3f}".format(self._Brain._X[i]) + ', '
        l.rstrip(', ')
        l += '\n'
        return l

    def WriteLine(self):
        for fileID in range(self._NumberOfFiles):
            l = self.ConstructLine(fileID)
            with open(self._FileNames[fileID],"a") as f: #in write mode
                f.write(l)

    def WriteParameters(self):
        with open(self._DataFolder+"/Parameters.json", 'w') as f:
            self._Parameters["tend"]            = self._Brain._tend
            self._Parameters["dt"]              = self._Brain._dt
            self._Parameters["ConnectionScale"] = self._Brain._ConnectionScale
            self._Parameters["NetworkDevel"]    = self._Brain._NetworkDevel
            self._Parameters["NeuronsPerFile"]  = self._NeuronsPerFile
            self._Parameters["NumberOfFiles"]   = self._NumberOfFiles
            self._Parameters["DataFolder"]      = self._DataFolder
            self._Parameters["NumberOfNeurons"] = self._Brain._NumberOfNeurons
            self._Parameters["SynapseLimit"]    = self._Brain._SynapseLimit
            json.dump(self._Brain._Params, f, indent=4, separators=(',', ': '))

    def WriteNetwork(self):
        data = {}
        data["Coordinate"] = {i:list(n) for i,n in enumerate(self._Brain._NeuronPosition)}
        data["Connections"] = {str(key):value for key,value in zip(self._Brain._SynapseWeight.keys(), self._Brain._SynapseWeight.values())}

        with open(self._DataFolder+"/Network.json", 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))

    def ReadFile(self, filename):
        filedata = []
        with open(filename, 'r') as f:
            for line in f:
                dataline = list(map(float,line.rstrip(', \n').lstrip('').split(',')))
                filedata.append(dataline)
        return filedata

    def ReadData(self):
        for fileID in range(self._NumberOfFiles):
            filedata=self.ReadFile(self._FileNames[fileID])
            if fileID == 0:
                self._FullData = [row[1:] for row in filedata]
                self._time = [row[0] for row in self._FullData]
            else:
                self._FullData = [Arow+Brow[1:] for Arow,Brow in zip(self._FullData,filedata)]
