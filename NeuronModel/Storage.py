import json
import os
import numpy as np
class Storage:
    def __init__(self, DataFolder, NeuronsPerFile, brain=None, NumberOfFiles=None, NumberOfNeurons=None):
        self._Brain          = brain
        self._DataFolder     = DataFolder
        if not os.path.exists(self._DataFolder):
            os.makedirs(self._DataFolder)
        #self._ParameterFile = kwargs["ParameterFile"]
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

        return cls(DataFolder, NeuronsPerFile=NeuronsPerFile, NumberOfFiles=NumberOfFiles, NumberOfNeurons=NumberOfNeurons, ParameterFile=name)

    def GetParams(self,NumberOfNeurons,NumberOfFiles,NeuronsPerFile):

        if self._Brain == None:
            with open(name) as data_file:
                data = json.load(self._ParameterFile)
            self._tend = data["tend"]
            self._dt   = data["dt"]
            self._ConnectionScale   = data["ConnectionScale"]
            self._SynapseLimit   = data["SynapseLimit"]
            self._NeuronsPerFile = NeuronsPerFile
            self._NumberOfNeurons = NumberOfNeurons
            self._NumberOfFiles   = NumberOfFiles
            self._Parameters      = {}

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
            json.dump(self._Brain._Params, f, indent=4, separators=(',', ': '))

    def ReadFile(self, filename):
        filedata = []
        with open(filename, 'r') as f:
            while True:
                line     = f.readline()
                dataline = list(map(float,line.rstrip(', \n').lstrip('').split(',')))
                filedata.append(dataline)
        return filedata

    def ReadData(self):
        for i,file in enumerate(self._FileNames):
            filedata=np.array(self.ReadFile())
            if i == 0:
                self._FullData = filedata
                time = [row[0] for row in self._FullData]
            else:
                self._FullData = [Arow+Brow[1:] for Arow,Brow in zip(self._FullData,filedata)]
