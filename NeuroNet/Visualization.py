from NeuroNet.Brain import *
class Visualization:

    def __init__(self, network, neurons, synapsecount):
        '''
            Visualization class
        '''
        self._Network = network
        self._Neurons = neurons
        self._NetworkEdgeWeightFactor = 1
        self._NodeLabels = {}
        self._SynapseCountHistory = synapsecount

    def PlotConnectivityProperties(self):

        #self._AverageConnectivity.append(nx.average_node_connectivity(self._Network))

        plt.subplot(131)
        self._DegreeDistribution = sorted(nx.degree(self._Network).values(),reverse=True)
        p=plt.loglog(self._DegreeDistribution,'-',marker='o')
        plt.setp(p,color='darkblue')

        plt.title("Degree rank plot")
        plt.ylabel("degree")
        plt.xlabel("rank")

        plt.subplot(132)
        plt.hist([n._SynapseCount for n in self._Neurons], 20,facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
        plt.title('Distribution of number of synapses formed')
        plt.xlabel('Number of synapses')
        plt.ylabel('Count')

        plt.subplot(133)
        weights = [self._Network.get_edge_data(n1,n2,default=0)['weight'] for n1,n2 in self._Network.edges()]
        plt.hist(weights, 20, facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
        plt.title('Distribution of edge weights')
        plt.xlabel('Edge weight')
        plt.ylabel('Count')

        plt.show()

    def DrawNetwork(self, edgelabels=False):
        warnings.filterwarnings("ignore")
        fig1 = plt.figure()
        ax = fig1.add_subplot(111, aspect='equal')
        plt.xlim((0,80))
        plt.ylim((0,80))


        for n in self._Neurons:
            self._NodeLabels[n]=n._ID
            if n._ActiveQ and n._ID != 0:
                nx.set_node_attributes(self._Network, 'color', {n:'#918b21'})

        pos = nx.get_node_attributes(self._Network,'pos')
        weights = [self._NetworkEdgeWeightFactor/u._SynapseLimit*self._Network[u][v]['weight'] for u,v in self._Network.edges()]
        nodecolors = list(nx.get_node_attributes(self._Network, 'color').values())

        nx.draw(self._Network, pos, node_size=110, width=weights)
        nx.draw_networkx_nodes(self._Network, pos, node_size=110,node_color=nodecolors)
        nx.draw_networkx_labels(self._Network, pos,  labels=self._NodeLabels,font_color=[1,1,1],font_family='Times',font_size=8)
        nx.draw_networkx_edges(self._Network, pos, edge_color='#2c8c7d', width=weights,arrows=True)
        if edgelabels:
            nx.draw_networkx_edge_labels(self._Network, pos, edge_labels=self._EdgeLabels, label_pos=0.5,font_size=6)

        # Background color to represent the experiment dish.
        ax.add_patch(patches.Rectangle((10, 0),60,10,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((70, 10),10,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((10, 70),60,10,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((0, 10),10,60,facecolor=[0.1,0.1,0.6],alpha=0.2))
        ax.add_patch(patches.Rectangle((10, 10),60,60,facecolor=[0.1,0.1,0.6],alpha=0.2))

        plt.show()

    def ComputeEigenValues(self,matrix="Laplacian"):
        if matrix=="Laplacian":
            L = nx.directed_laplacian_matrix(self._Network)
            eigs = np.linalg.eigvals(L)
        elif matrix=="Adjacency":
            eigs = sorted(nx.adjacency_spectrum(self._Network))
        return eigs

    def PlotEigenValues(self):
        plt.figure

        plt.subplot(121)
        eigs = self.ComputeEigenValues(matrix="Adjacency")
        reals = [np.real(n) for n in eigs]
        imag  = [np.imag(n) for n in eigs]
        plt.plot(reals,imag,'o')
        plt.grid(True)
        plt.title('Eigenvalues of Adjacency Matrix')
        plt.xlabel('real part')
        plt.ylabel('imaginary part')

        plt.subplot(122)
        eigs = self.ComputeEigenValues(matrix="Laplacian")
        reals = [np.real(n) for n in eigs]
        imag  = [np.imag(n) for n in eigs]
        plt.plot(reals,imag,'o')
        plt.grid(True)
        plt.title('Eigenvalues of Laplacian Matrix')
        plt.xlabel('real part')


        plt.show()

    def PlotDegreeDistribution(self):
        idegree = list(self._Network.in_degree().values())
        odegree = list(self._Network.out_degree().values())

        try:
            plt.subplot(131)
            plt.hist(idegree, max(idegree), facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
            plt.title('in-degree')
            plt.xlabel('degree')
            plt.ylabel('count')
            plt.subplot(132)
            plt.hist(odegree, max(odegree), facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
            plt.title('out-degree')
            plt.xlabel('degree')
            plt.ylabel('count')
            plt.subplot(133)
            plt.hist(self._DegreeDistribution, max(self._DegreeDistribution), facecolor='lightblue', alpha=0.75, edgecolor='darkblue')
            plt.title('total degree distribution')
            plt.xlabel('degree')
            plt.ylabel('count')
            plt.show()
        except:
            print("No connections at all")

    def PlotSynapseRank(self):
        p = plt.plot(self._Neurons[0]._Time,self._SynapseCountHistory)
        plt.setp(p, 'Color', [0.6,0.4,0.5], 'linewidth', 3)
        plt.grid(True)
        plt.ylabel("Count")
        plt.xlabel("Time")
        plt.title('Synapse Count Increase Over Time')
        plt.show()
