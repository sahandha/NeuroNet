#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import copy

class myClass():
    Comm = MPI.COMM_WORLD

    def __init__(self):
        self._mylist = np.ones(8)
        self._xp     = np.empty(int(len(self._mylist)/myClass.Comm.size))
        self.split()

    def split(self):
        cs = myClass.Comm.size
        s  = int(len(self._mylist)/cs)
        r = myClass.Comm.rank
        self._xp = copy.copy(self._mylist[s*r:s*(r+1)])
        #myClass.Comm.Scatter([self._mylist, MPI.DOUBLE], [self._xp, MPI.DOUBLE])

    def compute(self):
        r = myClass.Comm.rank
        self._xp = self._xp*r
        self.communicate()


    def communicate(self):
        myClass.Comm.Allgather( [self._xp, MPI.DOUBLE], [self._mylist, MPI.DOUBLE] )


    def printResult(self):
        r = myClass.Comm.rank
        print('from rank ',r,' I got: ', self._xp)
        print('from rank ',r,' I got: ', self._mylist)

if __name__=='__main__':
    c = myClass()
    c.compute()
    c.printResult()
