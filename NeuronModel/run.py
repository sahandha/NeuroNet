from NeuronModel import *
from Visualization import *
from Storage import *


C,gL,gCa,gK,VL,VCa,VK,V1,V2,V3,V4,phi=20,2.0,4.4,8,-60,120,-84,-1.2,18.0,2.0,30.0,0.04
C_v,gL_v,gCa_v,gK_v,VL_v,VCa_v,VK_v,V1_v,V2_v,V3_v,V4_v,phi_v=1,0,0,0,0,0,0,0,0,0,0,0

connectionscale = 40
N = 10

brain = NeuronModel(N=3, tend=1000, I=0, connectionscale=connectionscale, synapselimit=50, synapsestrengthlimit=60,
                    C=C,gL=gL,gCa=gCa,gK=gK,VL=VL,VCa=VCa,VK=VK,V1=V1,V2=V2,V3=V3,V4=V4,phi=phi,
                    I_v=0.1,C_v=C_v,gL_v=gL_v,gCa_v=gCa_v,gK_v=gK_v,VL_v=VL_v,VCa_v=VCa_v,VK_v=VK_v,V1_v=V1_v,V2_v=V2_v,V3_v=V3_v,V4_v=V4_v,phi_v=phi_v)

storage =  Storage(brain, '/Users/sahand/Research/NeuroNet/NeuronModel/Data', 10)

brain.SetStorage(storage)

brain.DevelopNetwork(120)
brain.Simulate(source='script')

#vis=Visualization(brain)

#neurons=[0,1,2,3]
#plt.rcParams["figure.figsize"] = (20,2*len(neurons))
#vis.PlotState(neurons=neurons)

#plt.rcParams["figure.figsize"] = (20,8)
#vis.PlotTimeFrequencyDot(plot_height=600, plot_width=1400, radius=1, cutoff=50, sorted=True)
