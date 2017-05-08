import sys, getopt
from NeuronModel import *
from Storage import *


def main(argv):

    # Defaults
    fromFile     = False
    fileName     = '/Users/sahand/Research/NeuroNet/Data/Parameters.json'
    outputFolder = '/Users/sahand/Research/NeuroNet/Data'

    connectionscale = 10
    N               = 80
    synapselimit    = 50
    NeuronPerFile   = 10
    NetworkDevel    = 120

    try:
        opts, args = getopt.getopt(argv,"hF:O:N:S:P:D:L:")
    except getopt.GetoptError:
        print(
        '''python3 run.py -options \n
        -h:   Help
        -F:   Specify the data file from which to build storage object.
        -O:   Data folder within which data is to be stored.
        -N:   Number of neurons
        -L:   Synapse limit
        -S:   Connection Scale
        -P:   Number of neurons per file.
        -D:   Network Development period.
        ''')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
            '''python3 run.py -options \n
            -h:   Help
            -F:   Specify the data file from which to build storage object.
            -O:   Data folder within which data is to be stored.
            -N:   Number of neurons
            -L:   Synapse limit
            -S:   Connection Scale
            -D:   Network development period
            -P:   Number of neurons per file.
            ''')
            sys.exit()
        elif opt in ("-F"):
            fromFile = True
            fileName = arg
        elif opt in ("-O"):
            outputFolder = arg
        elif opt in ("-N"):
            N = int(arg)
        elif opt in ("-L"):
            synapselimit = int(arg)
        elif opt in ("-S"):
            connectionscale = int(arg)
        elif opt in ("-D"):
            NetworkDevel = int(arg)
        elif opt in ("-P"):
            NeuronPerFile = int(arg)

    if fromFile:
        storage = Storage.FromFile(fileName)
        storage.ReadData()
        brain = NeuronModel(N=storage._NumberOfNeurons, connectionscale=storage._ConnectionScale, synapselimit=storage._SynapseLimit, synapsestrengthlimit=storage._SynapseLimit, **storage._Parameters)

    else:
        C,gL,gCa,gK,VL,VCa,VK,V1,V2,V3,V4,phi=20,2.0,4.4,8,-60,120,-84,-1.2,18.0,2.0,30.0,0.04
        C_v,gL_v,gCa_v,gK_v,VL_v,VCa_v,VK_v,V1_v,V2_v,V3_v,V4_v,phi_v=1,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01

        brain = NeuronModel(N=N, tend=2000, I=0, connectionscale=connectionscale, synapselimit=synapselimit, synapsestrengthlimit=synapselimit,
                        C=C,gL=gL,gCa=gCa,gK=gK,VL=VL,VCa=VCa,VK=VK,V1=V1,V2=V2,V3=V3,V4=V4,phi=phi,
                        I_v=0.1,C_v=C_v,gL_v=gL_v,gCa_v=gCa_v,gK_v=gK_v,VL_v=VL_v,VCa_v=VCa_v,VK_v=VK_v,V1_v=V1_v,V2_v=V2_v,V3_v=V3_v,V4_v=V4_v,phi_v=phi_v)
        brain.DevelopNetwork(NetworkDevel)

        storage =  Storage(outputFolder, NeuronPerFile, brain=brain)
        brain.SetStorage(storage)
        storage.WriteParameters()
        storage.WriteNetwork()
        brain.SetStorage(storage)

        brain.Simulate(source='script')


if __name__=='__main__':
    main(sys.argv[1:])
