#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:22:58 2023

@author: alexander
"""

import matplotlib.pyplot as pl
import fluidfoam as ff
from matplotlib import cm
import numpy as np
import os
import h5py


directory = 'channel_pisoFoamNN/'

timesteps = np.arange(70, 101, 1)

x, y, z = ff.readmesh(directory, structured=True)

u = np.ndarray((3, len(x), len(y[0]), len(z[0,0]), len(timesteps)))
#p = np.ndarray((len(x), len(y[0]), len(z[0,0]), len(timesteps)))
tau = np.ndarray((9, len(x), len(y[0]), len(z[0,0]), len(timesteps)))
tau_wall = np.ndarray((3, len(x), 1, len(z[0,0]), len(timesteps)))
nut = np.ndarray((len(x), len(y[0]), len(z[0,0]), len(timesteps)))

for i in range(len(timesteps)):
    timename = str(timesteps[i])
    print(timename)
    u[:, :, :, :, i] = ff.readvector(directory, timename, 'U', structured=True)
    #p[:, :, :, i] = ff.readscalar(directory, timename, 'p', structured=True)
    if timesteps[i] > 0:
        tau[:, :, :, :, i] = ff.readtensor(directory, timename, 'TauNN', structured=True)
        tau_wall[:, :, :, :, i] = ff.readvector(directory, timename, 'wallShearStress', structured=True, boundary='bottomWall')
        nut[:, :, :, i] = ff.readscalar(directory, timename, 'nut', structured=True)



filename = directory + 'simulation1.h5'
datafile = h5py.File(filename, 'w')
datafile.create_dataset('time', data=timesteps)
datafile.create_dataset('xcoor', data=x)
datafile.create_dataset('ycoor', data=y)
datafile.create_dataset('zcoor', data=z)
datafile.create_dataset('u', data=u)
datafile.create_dataset('nut', data=nut)
#datafile.create_dataset('p', data=p)
datafile.create_dataset('tau', data=tau)
datafile.create_dataset('tau_wall', data=tau_wall)
datafile.close()

