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


directory = 'channel_pisoFoamNN/'

timesteps = np.arange(1, 21, 1)

x, y, z = ff.readmesh(directory, structured=True)

u = np.ndarray((3, len(x), len(y[0]), len(z[0,0]), len(timesteps)))
p = np.ndarray((len(x), len(y[0]), len(z[0,0]), len(timesteps)))
tau = np.ndarray((9, len(x), len(y[0]), len(z[0,0]), len(timesteps)))


for i in range(len(timesteps)):
    timename = str(timesteps[i])
    print(timename)
    u[:, :, :, :, i] = ff.readvector(directory, timename, 'U', structured=True)
    p[:, :, :, i] = ff.readscalar(directory, timename, 'p', structured=True)
    if i > 0:
        tau[:, :, :, :, i] = ff.readtensor(directory, timename, 'TauNN', structured=True)


u_mean = np.mean(u, axis=(1,3,4))
p_mean = np.mean(p, axis=(0,2,3))
TauNN_mean = np.mean(tau, axis=(1,3,4))


fig2 = pl.figure(figsize=(10, 10))
sub = fig2.add_subplot(111)
sub.plot(TauNN_mean[0, :], y[0, :, 0], label='Tauxx')
sub.plot(TauNN_mean[4, :], y[0, :, 0], label='Tauyy')
sub.plot(TauNN_mean[8, :], y[0, :, 0], label='Tauzz')
sub.plot(TauNN_mean[1, :], y[0, :, 0], label='Tauxy')
sub.plot(TauNN_mean[2, :], y[0, :, 0], label='Tauxz')
sub.plot(TauNN_mean[5, :], y[0, :, 0], label='Tauyz')
sub.grid(True)
sub.set_xlabel('x')
sub.set_ylabel('TauNN')
pl.legend()
pl.show()

u_mean_space = np.mean(u, axis=(1,3))

fig3 = pl.figure(figsize=(10, 10))
sub = fig3.add_subplot(111)
sub.plot(u_mean[0,:], y[0, :, 0], label='time averaged profile')
sub.plot(u_mean_space[0,:,0], y[0, :, 0], label='average profile at t=1')
sub.plot(u_mean_space[0,:,4], y[0, :, 0], label='average profile at t=5')
sub.plot(u_mean_space[0,:,9], y[0, :, 0], label='average profile at t=10')
sub.plot(u_mean_space[0,:,14], y[0, :, 0], label='average profile at t=15')
sub.plot(u_mean_space[0,:,19], y[0, :, 0], label='average profile at t=20')
sub.grid(True)
sub.set_xlabel('x')
sub.set_ylabel('u')
pl.legend()
pl.show()