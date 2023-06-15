#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:22:58 2023

@author: alexander


The code reads the OpenFOAM simulation output data and saves it in a single HDF5 file
"""

#import libraries
import fluidfoam as ff
import numpy as np
import os
import h5py


directory = 'channel_pisoFoamNN/'                                           #define the simulation directory

timesteps = np.arange(70, 101, 1)                                           #define the timesteps in the solution

x, y, z = ff.readmesh(directory, structured=True)                           #read the mesh

#create arrays for the relevant variables of the size of the solution
u = np.ndarray((3, len(x), len(y[0]), len(z[0,0]), len(timesteps)))
tau = np.ndarray((9, len(x), len(y[0]), len(z[0,0]), len(timesteps)))
tau_wall = np.ndarray((3, len(x), 1, len(z[0,0]), len(timesteps)))
nut = np.ndarray((len(x), len(y[0]), len(z[0,0]), len(timesteps)))

#loop through the timesteps
#read the values of the relevant data using fluidfoam and input into the correct arrays
for i in range(len(timesteps)):
    timename = str(timesteps[i])
    u[:, :, :, :, i] = ff.readvector(directory, timename, 'U', structured=True)
    if timesteps[i] > 0:
        tau[:, :, :, :, i] = ff.readtensor(directory, timename, 'TauNN', structured=True)
        tau_wall[:, :, :, :, i] = ff.readvector(directory, timename, 'wallShearStress', structured=True, boundary='bottomWall')
        nut[:, :, :, i] = ff.readscalar(directory, timename, 'nut', structured=True)



#create the HDF5 file and save the arrays with their respective title
filename = directory + 'simulation1.h5'
datafile = h5py.File(filename, 'w')
datafile.create_dataset('time', data=timesteps)
datafile.create_dataset('xcoor', data=x)
datafile.create_dataset('ycoor', data=y)
datafile.create_dataset('zcoor', data=z)
datafile.create_dataset('u', data=u)
datafile.create_dataset('nut', data=nut)
datafile.create_dataset('tau', data=tau)
datafile.create_dataset('tau_wall', data=tau_wall)
datafile.close()

