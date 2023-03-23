# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 08:56:15 2022

@author: P27
"""

import h5py
import numpy as np
import matplotlib.pyplot as pl


nu = 5 * 10**(-5)
u_tau = 0.0499


file = h5py.File('channel_1b_processed.h5', 'r')
keys = list(file.keys())
u = np.array(file['Velocity_0001'])
xcoor = np.array(file['xcoor'])
ycoor = np.array(file['ycoor'])
zcoor = np.array(file['zcoor'])
filtered_ycoor = np.array(file['filtered_ycoor'])
ufilter = np.array(file['Velocity_Box_filter'])
file.close()

y_plus = ycoor * u_tau / nu
yfb_plus = filtered_ycoor * u_tau / nu
u_plus = u[0, :, :, :] / u_tau
ufb_plus = ufilter / u_tau



file = h5py.File('channel_1c_processed.h5', 'r')
filtered_ycoor = np.array(file['filtered_ycoor'])
ufilter = np.array(file['Velocity_Box_filter'])
file.close()

ufc_plus = ufilter / u_tau
yfc_plus = filtered_ycoor * u_tau / nu



file = h5py.File('channel_1d_processed.h5', 'r')
filtered_ycoor = np.array(file['filtered_ycoor'])
ufilter = np.array(file['Velocity_Box_filter'])
file.close()

ufd_plus = ufilter / u_tau
yfd_plus = filtered_ycoor * u_tau / nu



file = h5py.File('channel_1e_processed.h5', 'r')
filtered_ycoor = np.array(file['filtered_ycoor'])
ufilter = np.array(file['Velocity_Box_filter'])
file.close()

ufe_plus = ufilter / u_tau
yfe_plus = filtered_ycoor * u_tau / nu





y_plus = ycoor * u_tau / nu
yf1_plus = filtered_ycoor * u_tau / nu
u_plus = u[0, :, :, :] / u_tau
uf_plus = ufilter / u_tau

fig = pl.figure()
subplot = fig.add_subplot(111)
subplot.plot(y_plus, u_plus[26, :, 26], color = 'blue', label = 'DNS data fit')
#subplot.scatter(ycoor[ycoor < ycoor_clip], u[0, 0, :, 100][ycoor < ycoor_clip], color = 'red')
subplot.scatter(y_plus, u_plus[26, :, 26], color = 'red', label = 'DNS data')
subplot.scatter(yfb_plus, ufb_plus[0, 8, :, 8], color = 'darkgreen', marker = 'x', label = 'filtered DNS data, filter size = 8')
subplot.scatter(yfc_plus, ufc_plus[0, 4, :, 4], color = 'black', marker = '^', label = 'filtered DNS data, filter size = 16')
subplot.scatter(yfd_plus, ufd_plus[0, 2, :, 2], color = 'darkorange', marker = '<', label = 'filtered DNS data, filter size = 32')
subplot.scatter(yfe_plus, ufe_plus[0, 0, :, 0], color = 'magenta', marker = '>', label = 'filtered DNS data, filter size = 64')
subplot.set_xscale('log')
subplot.set_xlabel('$y^+$')
subplot.set_ylabel('$U^+$')
subplot.grid(True)
subplot.legend()

#[ycoor < 0.035]