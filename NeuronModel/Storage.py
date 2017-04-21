

class Storage:
    def __init__(self, brain, DataFolder, NeuronsPerFile):
        self._Brain         = brain
        self._DataFolder    = DataFolder
        self._NumberOfFiles = int(self._Brain._NumberOfNeurons/NeuronsPerFile)
        self._FileNames     = {}
        self.AssembleFileNames()

    def AssembleFileNames(self):
        for i in range(self._NumberOfFiles):
            name = self._DataFolder + "/data" + str(i).zfill(4) + ".dat"
            self._FileNames[i] = name

    def ConstructLine(self,fileID):
        l = "{:.3f}".format(self._Brain._t) + ', '
        for i in range(self._Brain._NumberOfNeurons):
                l += "{:.3f}".format(self._Brain._X[i]) + ', '
        l += '\n'
        return l

    def WriteLine(self):
        for fileID in range(self._NumberOfFiles):
            l = self.ConstructLine(fileID)
            with open(self._FileNames[fileID],"a") as f: #in write mode
                f.write(l)
