import matplotlib.pylab as plt
import matplotlib.patches as patches
import bokeh.plotting as bk
import networkx as nx
import numpy as np
import warnings
import igraph as ig
import os
import json
import glob
import itertools as it
import matplotlib.cm as cm


def FixPathsParallel(datafolder="/Users/sahand/Research/NeuroNet/Data/Sim5_0626201"):
    folders = os.listdir(datafolder)
    visfolder = datafolder+"/"+"Vis"
    if not os.path.exists(visfolder):
        os.makedirs(visfolder)
    with open(datafolder+"/Parameters.json", 'r') as file:
        data = json.load(file)
        data["DataFolder"]=datafolder
    with open(datafolder+"/Parameters.json", 'w') as file:
        json.dump(data, file, indent=4, separators=(',', ': '))

class DictTable(dict):
    # Overridden dict class which takes a dict in the form {'a': 2, 'b': 3},
    # and renders an HTML Table in IPython Notebook.
    def _repr_html_(self):
        html = ['<table align="left" width=100%>']
        html.append('<tr>')
        html.append('<th align="left">Paramater Name</th>')
        html.append('<th align="left">Paramater Value</th>')
        html.append("<tr>")
        for key, value in iter(self.items()):
            html.append("<tr>")
            html.append("<td>{0}</td>".format(str(key)))
            html.append("<td>{0}</td>".format(str(value)))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

def ReadFileIndividualNeuron(filename, column=0):
    filedata = []
    with open(filename, 'r') as f:
        for line in f:
            ld = []
            dataline = line.rstrip(', \n').lstrip(' ').split(',')
            filedata.append(np.array([dataline[0],dataline[column+1]])) # +1 is here becuase the first column is always time.
    return np.array(filedata)

def VisTimeSeries(datafolder, neurons=[0],output='Save'):

    with open(datafolder+"/Parameters.json") as data_file:
            metadata = json.load(data_file)
    p = metadata["NeuronsPerFile"]

    filenumbers = []
    datacolumns = []
    for n in neurons:
        filenumbers.append(int(n/p))
        datacolumns.append(n%p)

    datafiles = glob.glob(datafolder+"/*.dat")

    files = [datafiles[n] for n in filenumbers]
    fig = plt.figure(figsize=(25,2*len(files)))

    for i, file in enumerate(files):
        ax = fig.add_subplot(len(files),1,i+1)
        data = ReadFileIndividualNeuron(file,column=datacolumns[i])
        p=plt.plot(data[:,0], data[:,1], rasterized=True)
        plt.ylim([-90,90])
        plt.setp(p,'Color', [0.7,0.3,0.3], 'linewidth', 3)
        plt.ylabel('{}'.format(neurons[i]))
        plt.grid(True)
    plt.xlabel('Time')
    if output=='Show':
        plt.show()
    else:
        plt.savefig(datafolder +'/Vis/IndividualTimeSeries.png', bbox_inches='tight')

def SortNeurons(datafolder):
    d = []
    neurontoindex = {}
    coordinates = np.genfromtxt(datafolder+"/Positions.csv", delimiter=',')
    for nxy in coordinates:
        d.append((nxy[0],nxy[1]**2+nxy[2]**2))
    sl = sorted(d, key=lambda x: x[1])
    SortedNeurons = [int(x[0]) for x in sl]
    for i,n in enumerate(SortedNeurons):
        neurontoindex[n] = i

    return neurontoindex

def VisAdjacencyMatrix(datafolder, sort=False, numberoffiles=2, output='Save',figsize=(25,25)):

    with open(datafolder+"/Parameters.json") as data_file:
            metadata = json.load(data_file)

    networkfiles = glob.glob(datafolder+"/Network/*.csv")

    if numberoffiles=='All':
        files = networkfiles
    else:
        files = networkfiles[0:numberoffiles]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)

    if sort:
        SortedNeruons = SortNeurons(datafolder)

    for i,file in enumerate(files):
        data = np.genfromtxt(file, delimiter=',')
        if sort:
            xdata = np.array([SortedNeruons[d] for d in data[:,0]])
            ydata = np.array([SortedNeruons[d] for d in data[:,1]])
        else:
            xdata = data[:,0]
            ydata = data[:,1]
        weights = data[:,2]
        plt.scatter(xdata, ydata, c=weights, s=5, cmap='Blues', rasterized=True)
    if sort:
        ax.set_xticklabels(SortedNeruons.keys())
        ax.set_yticklabels(SortedNeruons.keys())
    plt.colorbar()
    if output=='Show':
        plt.show()
    else:
        plt.savefig(datafolder +'/Vis/AdjacencyMatrix.png', bbox_inches='tight')

def ReadFile(filename, filenumber=0, perfile=200, threashold=40):
    filedata = []
    with open(filename, 'r') as f:
        for line in f:
            ld = []
            dataline = line.rstrip(', \n').lstrip(' ').split(',')
            for i,data in enumerate(dataline[1:]):
                if data == 'none':
                    data = 0
                if float(data) > threashold:
                    ld.append(np.array([float(dataline[0]), perfile*filenumber + int(i)]))

            if len(ld)>0:
                filedata.append(ld)
    return filedata


def VisTimeFreq(datafolder, cuttoff=50, output = 'Save',figsize=(25,10)):

    with open(datafolder+"/Parameters.json") as data_file:
            metadata = json.load(data_file)

    datafiles = glob.glob(datafolder+"/*.dat")
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    for i,file in enumerate(datafiles):
        f = ReadFile(file,filenumber=i,perfile=metadata["NeuronsPerFile"],threashold=cuttoff)
    
        if len(f)>0:
            data = np.array([item for sublist in f for item in sublist])
            ax.scatter(data[:,0], data[:,1], c=[0.7]*3, s=2, rasterized=True)
    if output=='Show':
        plt.show()
    else:
        plt.savefig(datafolder +'/Vis/TimeFreq.png', bbox_inches='tight')


class Visualization:
    def __init__(self,brain, FigNume=1):
        self._Brain   = brain
        self._NumberOfNeurons = self._Brain._NumberOfNeurons
        self._Neurons = list(range(self._NumberOfNeurons))
        self._FigNum  = FigNume
        #self.ConstructNetwork()
        self.SortNeurons()

    def ConstructNetwork(self):
        nodeData = [(0,{'color':'#db203c','pos':self._Brain._NeuronPosition[0]})]+[(n,{'color':'#1941d3','pos':self._Brain._NeuronPosition[n]}) for n in range(1,self._NumberOfNeurons)]
        edgeData=[(*p,{'weight':self._Brain._SynapseWeight[p]/self._Brain._SynapseStrengthLimit}) for p in self._Brain._SynapseWeight.keys() if self._Brain._SynapseWeight[p]>0]

        self._Network= nx.DiGraph()
        self._Network.add_nodes_from(nodeData)
        self._Network.add_edges_from(edgeData)
        self._DegreeDistribution = sorted(nx.degree(self._Network).values(),reverse=True)
    def ConstructNetwork_igraph(self, linewidth):

        ymax = max([self._Brain._NeuronPosition[n][1] for n in range(0,self._NumberOfNeurons)])
        self._g = ig.Graph(directed=True)
        self._g.add_vertices(self._NumberOfNeurons)
        self._g.vs["label"] = list(range(self._NumberOfNeurons))
        self._g.vs['x'] = [self._Brain._NeuronPosition[n][0] for n in range(0,self._NumberOfNeurons)]
        self._g.vs['y'] = [ymax-self._Brain._NeuronPosition[n][1] for n in range(0,self._NumberOfNeurons)]
        nodeData  = [p for p in self._Brain._SynapseWeight.keys() if self._Brain._SynapseWeight[p]>0]
        self._g.add_edges(nodeData)
        self._g.es["weight"] = [self._Brain._SynapseWeight[p]/self._Brain._SynapseStrengthLimit for p in self._Brain._SynapseWeight.keys() if self._Brain._SynapseWeight[p]>0]

        self._vis_style = {}
        outdegree = self._g.outdegree()
        self._vis_style["vertex_size"] = [x/max(outdegree)*10+5 for x in outdegree]
        self._vis_style["edge_curved"] = False
        self._vis_style["weights"] = self._g.es["weight"]
        self._vis_style["edge_width"] = [linewidth*e for e in self._g.es["weight"]]
        self._vis_style["arrow_size"] = [e for e in self._g.es["weight"]]
        self._vis_style["edge_curved"] = False
        self._vis_style['vertex_color'] = 'blue'
        self._vis_style["vertex_label_color"] = 'white'

    def PlotState(self,neurons=[0], render="Display",figsize=(30,15)):
        self._FigNum += 1
        plt.figure(self._FigNum,figsize=figsize)
        for idx, n in enumerate(neurons):
            plt.subplot(len(neurons),1,idx+1)
            p = plt.plot(self._Brain._Time,self._Brain._VV[:,n])
            plt.ylim( (-90, 80) )
            plt.setp(p, 'Color', [0.7,0.3,0.3], 'linewidth', 3)
            plt.grid(True)
            plt.xlabel("Time")
            plt.ylabel("Neuron {}".format(n))
        if render == "Display":
            plt.show()
        else:
            plt.savefig(render, bbox_inches='tight')

    def DrawNetwork(self, edgelabels=False, render="Display"):
        warnings.filterwarnings("ignore")
        self._FigNum += 1
        fig1=plt.figure(self._FigNum,figsize=(25,25))
        ax = fig1.add_subplot(111, aspect='equal')
        plt.xlim((0,80))
        plt.ylim((0,80))
        #for n in self._Neurons:
        #    self._NodeLabels[n]=n._ID
        #    if n._ActiveQ and n._ID != 0:
        #        nx.set_node_attributes(self._Network, 'color', {n:'#918b21'})
        pos = nx.get_node_attributes(self._Network,'pos')
        weightcap = max(self._Brain._SynapseWeight.values())
        weights = [self._Network[u][v]['weight'] for u,v in self._Network.edges()]
        nodecolors = list(nx.get_node_attributes(self._Network, 'color').values())
        nx.draw(self._Network, pos, node_size=110, width=weights)
        nx.draw_networkx_nodes(self._Network, pos, node_size=110,node_color=nodecolors)
        nx.draw_networkx_labels(self._Network, pos,  labels={n:n for n in self._Neurons},font_color=[1,1,1],font_family='Times',font_size=8)
        nx.draw_networkx_edges(self._Network, pos, edge_color='#2c8c7d', width=weights,arrows=True)

        # Background color to represent the experiment dish.
        ax.add_patch(patches.Rectangle((10, 0),60,10,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((70, 10),10,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((10, 70),60,10,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((0, 10),10,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((10, 10),60,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        if render == "Display":
            plt.show()
        else:
            plt.savefig(render, bbox_inches='tight')

    def SortNeurons(self):
        d = []
        for n in range(self._NumberOfNeurons):
            d.append((n,self._Brain._NeuronPosition[n][0]**2+self._Brain._NeuronPosition[n][1]**2))
        sl = sorted(d, key=lambda x: x[1])
        self._SortedNeurons = [x[0] for x in sl]

    def PlotTimeFrequencyDot(self,cutoff=20,radius=2,plot_height=100,plot_width=300, sorted=False, render="Display"):
        time = self._Brain._Time
        if sorted:
            #neurons = [row[0] for row in self._SortedNeurons]
            neurons = self._SortedNeurons
            title = "Firing Pattern of Neurons Sorted by Distance"
            factors = [str(n) for n in self._SortedNeurons]
        else:
            neurons = self._Neurons
            title = "Firing Pattern of Neurons"
            factors = [str(n) for n in self._Neurons]
        datax = np.array([])
        datay = np.array([])
        for i,n in enumerate(neurons):
            V = self._Brain._VV[:,n]
            d = time[V>cutoff]
            datax=np.append(datax, d)
            datay=np.append(datay, (i+1)*np.ones_like(d))
        p = bk.figure(title=title, y_range=factors, x_axis_label='Time', y_axis_label='Neurons',x_range=(0,time[-1]),plot_height=plot_height,plot_width=plot_width)
        p.circle(datax,datay,radius=radius, fill_color='darkblue', line_color='darkblue')
        if render == "Display":
            bk.show(p)
        else:
            bk.output_file(render)
            bk.save(p)

    def PlotTimeFrequencyDotPLT(self,cutoff=20, sorted=False, render="Display"):
        time = self._Brain._Time
        if sorted:
            #neurons = [row[0] for row in self._SortedNeurons]
            neurons = self._SortedNeurons
            title = "Firing Pattern of Neurons Sorted by Distance"
            factors = [str(n) for n in self._SortedNeurons]
        else:
            neurons = self._Neurons
            title = "Firing Pattern of Neurons"
            factors = [str(n) for n in self._Neurons]
        datax = np.array([])
        datay = np.array([])
        for i,n in enumerate(neurons):
            V = self._Brain._VV[:,n]
            d = time[V>cutoff]
            datax=np.append(datax, d)
            datay=np.append(datay, (i+1)*np.ones_like(d))

        self._FigNum += 1
        fig = plt.figure(self._FigNum,figsize=(30,15))
        ax = fig.add_subplot(111)

        p = plt.scatter(datax,datay,s=1)
        ax.set_aspect(0.5)
        if render == "Display":
            plt.show()
        else:
            plt.savefig(render, bbox_inches='tight')

    def PlotTimeFrequency(self, render="Display"):
        self._FigNum += 1
        plt.figure(self._FigNum,figsize=(30,15))
        time = self._Brain._Time
        data = np.zeros((self._NumberOfNeurons,len(time)))
        for i,n in enumerate(self._Neurons):
            for j,t in enumerate(time):
                data[i,j] = self._Brain._VV[j,n]
        ax = plt.subplot(1,1,1)
        p=ax.pcolorfast(time,range(len(self._Neurons)),data)
        plt.colorbar(p)
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Neurons')
        if render == "Display":
            plt.show()
        else:
            plt.savefig(render, bbox_inches='tight')

    def PlotAdjacencyMatrix(self, render="Display"):
        self._FigNum += 1
        fig = plt.figure(self._FigNum,figsize=(30,15))

        M = nx.to_numpy_matrix(self._Network)
        ax1 = plt.subplot(131)
        p=ax1.pcolorfast(M,cmap='Blues')
        plt.colorbar(p)
        plt.title('Unordered')


        ax2 = plt.subplot(132)
        p=ax2.pcolorfast(self.ComputeSortedAdjacencyMatrix(),cmap='Blues')
        ax2.set_xticks(self._SortedNeurons)
        ax2.set_yticks(self._SortedNeurons)
        plt.colorbar(p)
        plt.title('Ordered')


        ax3 = plt.subplot(133)
        H=np.array(np.ndarray.flatten(M))
        plt.hist(H[0],20)
        plt.title('Weight Distribution')
        #plt.tight_layout(pad=0.5)
        if render == "Display":
            plt.show()
        else:
            plt.savefig(render, bbox_inches='tight')

    def ComputeSortedAdjacencyMatrix(self):
        M = np.array([np.array([self._Brain._SynapseWeight[(n1,n2)]/self._Brain._SynapseStrengthLimit for n2 in self._SortedNeurons]) for n1 in self._SortedNeurons])
        return M

    def PlotConnectivityProperties(self, render="Display"):
        #self._AverageConnectivity.append(nx.average_node_connectivity(self._Network))
        self._FigNum += 1
        plt.figure(self._FigNum,figsize=(30,15))

        plt.subplot(131)
        #self._DegreeDistribution = sorted(nx.degree(self._Network).values(),reverse=True)
        p=plt.loglog(self._DegreeDistribution,'-',marker='o')
        plt.setp(p,color='darkblue')
        plt.title("Degree rank plot")
        plt.ylabel("degree")
        plt.xlabel("rank")
        plt.subplot(132)
        #self.GetSynapseCount()
        plt.hist(self._Brain._SynapseCount, 20,facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
        plt.title('Distribution of number of synapses formed')
        plt.xlabel('Number of synapses')
        plt.ylabel('Count')
        plt.xlim( (0, self._Brain._SynapseLimit) )
        plt.subplot(133)
        weights = list(self._Brain._SynapseWeight.values())#[self._Network.get_edge_data(n1,n2,default=0)['weight'] for n1,n2 in self._Network.edges()]
        plt.hist(weights, 20, facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
        plt.title('Distribution of edge weights')
        plt.xlabel('Edge weight')
        plt.ylabel('Count')
        plt.xlim( (0, self._Brain._SynapseStrengthLimit) )

        #plt.tight_layout(pad=0.5)
        if render == "Display":
            plt.show()
        else:
            plt.savefig(render, bbox_inches='tight')

    def PlotDegreeDistribution(self, render="Display"):
        idegree = list(self._Network.in_degree().values())
        odegree = list(self._Network.out_degree().values())
        self._FigNum += 1
        plt.figure(self._FigNum, figsize=(30,15))
        if True:
            plt.subplot(131)
            plt.hist(idegree, int(max(idegree)/2), facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
            plt.title('in-degree')
            plt.xlabel('degree')
            plt.ylabel('count')
            plt.subplot(132)
            plt.hist(odegree, int(max(odegree)/2), facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
            plt.title('out-degree')
            plt.xlabel('degree')
            plt.ylabel('count')
            plt.subplot(133)
            plt.hist(self._DegreeDistribution, int(max(self._DegreeDistribution)/2), facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
            plt.title('total degree distribution')
            plt.xlabel('degree')
            plt.ylabel('count')
            #plt.tight_layout(pad=0.5)
            if render == "Display":
                plt.show()
            else:
                plt.savefig(render, bbox_inches='tight')
        else:
            print("No connections at all")

    def GetSynapseCount(self):
        self._SynapseCount = []
        for n in self._Neurons:
            self._Brain_SynapseCount
            self._SynapseCount.append(sum([self._Brain._SynapseWeight[(n,i)] for i in self._Neurons]))

    def ClearPlot(self):
        plt.cla()
        plt.clf()
        plt.close()
