import Visualization as vis
import json
import os
import numpy as np


simfolder='/u/eot/hariria2/scratch/Parallel/08-27-2017'

files = os.listdir(simfolder)
simpaths = []
for f in files:
    simpaths.append(simfolder+'/'+f)

for simpath in simpaths:
    print(simpath)
    vis.FixPathsParallel(simpath)
    vis.VisTimeSeries(simpath, neurons=[0,10,20,50,100,300,500],output='Save')
    vis.VisTimeFreq(simpath, cuttoff=60, output = 'Save')
    vis.VisAdjacencyMatrix(simpath, sort=True, output='Save')
