import json
import os
class Storage:
    def __init__(self, DataFolder, NeuronsPerFile, brain=None, NumberOfFiles=None, NumberOfNeurons=None):
        self._Brain          = brain
        self._DataFolder     = DataFolder
        if not os.path.exists(self._DataFolder):
            os.makedirs(self._DataFolder)
        self.GetParams(NumberOfNeurons,NumberOfFiles,NeuronsPerFile)
        self._FileNames={}
        self._FullData=[]
        self.AssembleFileNames()


    @classmethod
    def FromFile(cls, name):
        with open(name) as data_file:
            data = json.load(data_file)
        NeuronsPerFile  = data["Neurons Per File"]
        DataFolder      = data["Data Folder"]
        NumberOfFiles   = data["Number of Files"]
        NumberOfNeurons = data["Number of Neurons"]

        return cls(DataFolder, NeuronsPerFile=NeuronsPerFile, NumberOfFiles=NumberOfFiles, NumberOfNeurons=NumberOfNeurons)

    def GetParams(self,NumberOfNeurons,NumberOfFiles,NeuronsPerFile):
        if self._Brain == None:
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
        l += '\n'
        return l

    def WriteLine(self):
        for fileID in range(self._NumberOfFiles):
            l = self.ConstructLine(fileID)
            with open(self._FileNames[fileID],"a") as f: #in write mode
                f.write(l)

    def WriteParameters(self):
        with open(self._DataFolder+"/Parameters.json", 'w') as f:
            self._Parameters["ConnectionScale"] = self._Brain._ConnectionScale
            self._Parameters["NetworkDevel"]    = self._Brain._NetworkDevel
            self._Parameters["NeuronsPerFile"]  = self._NeuronsPerFile
            self._Parameters["NumberOfFiles"]   = self._NumberOfFiles
            self._Parameters["DataFolder"]      = self._DataFolder
            self._Parameters["NumberOfNeurons"] = self._Brain._NumberOfNeurons
            json.dump(self._Brain._Params, f, indent=4, separators=(',', ': '))

    #def ReadData(self):
    #    for file in self._FileNames[]
