class Synapse:
    def __init__(self, ID, upstream, downstream):
        self._ID   = ID
        self._SynapticStrength = .1
        self.SetParentNeuron(upstream)
        self.SetChildNeuron(downstream)

    def SetParentNeuron(self, upstream):
        self._ParentNeuron = upstream
    def SetChildNeuron(self, downstream):
        self._ChildNeuron = downstream

    def Update(self):
        self._ChildNeuron.setInput(self._SynapticStrength*self._ParentNeuron.getW())
