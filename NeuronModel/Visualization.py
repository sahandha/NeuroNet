import matplotlib.pylab as plt
import matplotlib.patches as patches
import bokeh.plotting as bk
import networkx as nx
import numpy as np
import warnings
import igraph as ig
#import cairo

class Visualization:
    def __init__(self,brain):
        self._Brain   = brain
        self._NumberOfNeurons = self._Brain._NumberOfNeurons
        self._Neurons = list(range(self._NumberOfNeurons))
        self._FigNum  = 0
        self.ConstructNetwork()
        self.SortNeurons()

    def ConstructNetwork(self):
        nodeData = [(0,{'color':'#db203c','pos':self._Brain._NeuronPosition[0]})]+[(n,{'color':'#1941d3','pos':self._Brain._NeuronPosition[n]}) for n in range(1,self._NumberOfNeurons)]
        edgeData=[(*p,{'weight':self._Brain._SynapseWeight[p]/self._Brain._SynapseStrengthLimit}) for p in self._Brain._SynapseWeight.keys() if self._Brain._SynapseWeight[p]>0]

        self._Network= nx.DiGraph()
        self._Network.add_nodes_from(nodeData)
        self._Network.add_edges_from(edgeData)

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

    def PlotState(self,neurons=[0], render="Display"):
        plt.figure(self._FigNum)
        self._FigNum += 1
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
        fig1=plt.figure(self._FigNum)
        self._FigNum += 1
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
        fig = plt.figure(self._FigNum)
        ax = fig.add_subplot(111)

        p = plt.scatter(datax,datay)
        ax.set_aspect(0.5)
        if render == "Display":
            plt.show()
        else:
            plt.savefig(render, bbox_inches='tight')

    def PlotTimeFrequency(self, render="Display"):
        self._FigNum += 1
        plt.figure(self._FigNum)
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
        plt.figure(self._FigNum)
        self._FigNum += 1
        fig = plt.figure(self._FigNum,figsize=(15,4))

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
        plt.figure(self._FigNum,figsize=(15,4))

        plt.subplot(131)
        self._DegreeDistribution = sorted(nx.degree(self._Network).values(),reverse=True)
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
        plt.figure(self._FigNum, figsize=(15,4))
        self._FigNum += 1
        try:
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
        except:
            print("No connections at all")

    def GetSynapseCount(self):
        self._SynapseCount = []
        for n in self._Neurons:
            self._Brain_SynapseCount
            self._SynapseCount.append(sum([self._Brain._SynapseWeight[(n,i)] for i in self._Neurons]))
