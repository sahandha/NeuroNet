import csv
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import filtfilt, butter

class DataImport:

    def __init__(self, datafolder='../Data/', filename='data.dat'):
        self._t          = np.array([])
        self._x          = np.array([])
        self._DataFolder = datafolder
        self._FileName   = filename
        self._CellNumber = self._FileName.split('_')[-1].split('.')[0]

    def Read(self):
        with open(self._DataFolder+self._FileName) as f:
            reader = csv.reader(f, delimiter="\t")
            data=[list(map(float,row)) for row in reader]

        self._t = np.array([d[0 ] for d in data])
        self._x = np.array([d[1:] for d in data])
        self._DataSize   = len(self._t)
        self._dt         = self._t[ 1] - self._t[0]
        self._TimeWindow = self._t[-1] - self._t[0]

    def QuickView(self):
        plt.plot(self._t, self._x,'k')
        plt.title('Data for cell number {}.'.format(self._CellNumber))
        plt.xlabel(r'time (seconds)')
        plt.ylabel(r'potential ($\mu$V)')
        plt.show()

    def ZoomPlot(self, t1=400,t2=500):
        indx1 = int(t1/(1000*self._dt))
        indx2 = int(t2/(1000*self._dt))

        plt.plot(self._t[indx1:indx2], self._x[indx1:indx2],'k')
        plt.title('Data for cell number {}.'.format(self._CellNumber))
        plt.xlabel(r'time (seconds)')
        plt.ylabel(r'potential ($\mu$V)')
        plt.show()

    def ButterWorthFilter(self,order=3, cutoff=0.02):
        b, a = butter(order, cutoff)
        for i in range(self._x.shape[1]):
            self._x[:,i] = filtfilt(b,a,self._x[:,i])
