import sys, getopt
from NeuronModel import *
from Storage import *
import getpass
import os
import datetime as dt
import glob
from time import time


def getOutputFolder(postfix):
    user = getpass.getuser()
    date = dt.datetime.today().strftime("%m-%d-%Y")
    ver  = 0
    if user == "sahand":
        dataFolder = "/Users/sahand/Research/NeuroNet/Data"
    elif user == "hariria2":
        dataFolder = "/u/eot/hariria2/scratch/Parallel"
    else:
        print("Input data directory: ")
        dataFolder = input()
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    if not os.path.exists(dataFolder+"/"+date):
        os.makedirs(dataFolder+"/"+date)
    while len(glob.glob(dataFolder+"/"+date+"/"+"Sim"+str(ver)+"*"))>0:
        ver += 1

    os.makedirs(dataFolder+"/"+date+"/"+"Sim"+str(ver)+"_"+postfix)
    os.makedirs(dataFolder+"/"+date+"/"+"Sim"+str(ver)+"_"+postfix+"/"+"Network")
    os.makedirs("./Sim"+str(ver)+"_"+postfix)
    return (dataFolder+"/"+date+"/"+"Sim"+str(ver)+"_"+postfix, "./Sim"+str(ver)+"_"+postfix)



def main(argv):

    # Defaults
    fromFile     = False
    fileName     = '/Users/sahand/Research/NeuroNet/Data/Parameters.json'
    outputFolder = '/Users/sahand/Research/NeuroNet/Data'

    connectionscale = 40
    N               = 50
    synapselimit    = 50
    NeuronPerFile   = 10
    NetworkDevel    = 120
    JobID           = 0
    Tend            = 10

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
        -J:   Job ID assigned by BW
        -T:   Total time of simulation
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
            -J:   Job ID assigned by BW
            -T:   Total time of simulation
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
        elif opt in ("-J"):
            JobID = arg
        elif opt in ("-T"):
            Tend  = int(arg)

    (outputFolder, reportFolder) = getOutputFolder('N'+str(N)+'_L'+str(synapselimit)+'_S'+str(connectionscale)+'_D'+str(NetworkDevel)+'_T'+str(Tend))

    if fromFile:
        storage = Storage.FromFile(fileName)
        storage.ReadData()
        brain = NeuronModel(N=storage._NumberOfNeurons, connectionscale=storage._ConnectionScale, synapselimit=storage._SynapseLimit, synapsestrengthlimit=storage._SynapseLimit, **storage._Parameters)

    else:
        C,gL,gCa,gK,VL,VCa,VK,V1,V2,V3,V4,phi=20,2.0,4.4,8,-60,120,-84,-1.2,18.0,2.0,30.0,0.04
        C_v,gL_v,gCa_v,gK_v,VL_v,VCa_v,VK_v,V1_v,V2_v,V3_v,V4_v,phi_v=1,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01

        brain = NeuronModel(N=N, tend=2000, I=0, connectionscale=connectionscale, synapselimit=synapselimit, synapsestrengthlimit=synapselimit,networkdevel=NetworkDevel,
                        C=C,gL=gL,gCa=gCa,gK=gK,VL=VL,VCa=VCa,VK=VK,V1=V1,V2=V2,V3=V3,V4=V4,phi=phi,
                        I_v=0.1,C_v=C_v,gL_v=gL_v,gCa_v=gCa_v,gK_v=gK_v,VL_v=VL_v,VCa_v=VCa_v,VK_v=VK_v,V1_v=V1_v,V2_v=V2_v,V3_v=V3_v,V4_v=V4_v,phi_v=phi_v)


        storage =  Storage(outputFolder, NeuronPerFile, brain=brain)
        brain.SetStorage(storage)
        storage.WriteParameters()
        storage.WritePositions()
        #storage.WriteNetwork()
        #brain.SetStorage(storage)
        brain.DevelopNetwork()
        brain.Simulate(source='script')


if __name__=='__main__':
    main(sys.argv[1:])
