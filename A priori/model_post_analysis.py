# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 09:37:24 2022

@author: P27
"""



# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import cm
import scipy.ndimage
from scipy import signal
import h5py
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from datetime import timedelta
import json




save = False
NN_name = 'NN_18'

filetype = 'channel'


test_files = ['channel_30b_processed.h5',]
#              'channel_30c_processed.h5',
#              'channel_30d_processed.h5',
#              'channel_30e_processed.h5']

n_epochs = 500
activation_function = 'tanh'
layer_MLPs = [24, 12, 24, 6]
loss_function = 'mse'
learning_rate = 0.001

image_resolution = 31


C = 0.1



directory = str(NN_name) + '/'

filename = directory + 'scaling_parameters.h5'
scaling_params = h5py.File(filename, 'r')

ubox_scale = np.array(scaling_params['ubox_scale'])
ubox_min = np.array(scaling_params['ubox_min'])
filtered_grad_scale = np.array(scaling_params['filtered_gradient_scale'])
filtered_grad_min = np.array(scaling_params['filtered_gradient_min'])
strain_tensor_scale = np.array(scaling_params['strain_tensor_scale'])
strain_tensor_min = np.array(scaling_params['strain_tensor_min'])
yplus_scale = np.array(scaling_params['yplus_scale'])
yplus_min = np.array(scaling_params['yplus_min'])
Output_scale = np.array(scaling_params['output_scale'])
Output_min = np.array(scaling_params['output_min'])




def correlation_coefficient(test_output, predicted_output):
    
    yt = test_output
    yp = predicted_output
    
    N = tf.cast(len(yt), tf.float32)
    
    num = N * tf.reduce_sum(tf.multiply(yt, yp), axis=0) - tf.multiply(tf.reduce_sum(yt, axis=0), tf.reduce_sum(yp, axis=0))
    
    den1 = N * tf.reduce_sum(tf.square(yt), axis=0) - tf.square(tf.reduce_sum(yt, axis=0))
    den2 = N * tf.reduce_sum(tf.square(yp), axis=0) - tf.square(tf.reduce_sum(yp, axis=0))
    
    den = tf.sqrt(tf.multiply(den1, den2))
    
    r = tf.divide(num, den)
    
    return r


def cc_0(y_true, y_pred):
    r = correlation_coefficient(y_true, y_pred)
    return r[0]

def cc_1(y_true, y_pred):
    r = correlation_coefficient(y_true, y_pred)
    return r[1]

def cc_2(y_true, y_pred):
    r = correlation_coefficient(y_true, y_pred)
    return r[2]

def cc_3(y_true, y_pred):
    r = correlation_coefficient(y_true, y_pred)
    return r[3]

def cc_4(y_true, y_pred):
    r = correlation_coefficient(y_true, y_pred)
    return r[4]

def cc_5(y_true, y_pred):
    r = correlation_coefficient(y_true, y_pred)
    return r[5]





model_name = directory + 'model.h5'
model = tf.keras.models.load_model(model_name, custom_objects={'cc_0':cc_0, 'cc_1':cc_1, 'cc_2':cc_2, 'cc_3':cc_3, 'cc_4':cc_4, 'cc_5':cc_5})

history_dir = directory + 'model_history.json'
f = open(history_dir)
history = json.load(f)
f.close()





















def shape_test_data (file, ubox_scale, filtered_grad_scale, strain_tensor_scale, yplus_scale, Output_scale, ubox_min, filtered_grad_min, strain_tensor_min, yplus_min, Output_min):
    
    """
    Read the data from the hdf5 files and divide into input and output arrays.
    These contain information corresponding to a single spatial data point.
    The function takes the type of flow analysed and the number of data cubes.
    """
        

    filename = file
    preprocessed_data = h5py.File(filename, 'r')
    ubox = np.array(preprocessed_data['Velocity_Box_filter'])
    
    array_size = (len(ubox[0]))**3
    
    datacube_size = len(ubox[0])
    
    #create arrays for the single data types to be extracted
    ubox_train = np.ndarray((array_size, 3))
    filtered_gradient = np.ndarray((array_size, 9))
    strain_tensor_train = np.ndarray((array_size, 6))
#    SGS_frac = np.ndarray((array_size, 1))
    Filter_size = np.ndarray((array_size, 1))
    yplus = np.ndarray((array_size, 1))
    Input = []
    Output = np.ndarray((array_size, 6))
    
    
    counter = 0
    
    
        
    #read the file
    filename = file
    preprocessed_data = h5py.File(filename, 'r')
    ubox = np.array(preprocessed_data['Velocity_Box_filter'])
    filtered_ugradient = np.array(preprocessed_data['Gradient_of_filtered_velocity'])
    strain_tensor = np.array(preprocessed_data['Strain_tensor'])
    residual_stress = np.array(preprocessed_data['Residual_stress'])
    frac = np.array(preprocessed_data['Fraction_of_subgrid_scale_energy'])
    filter_size = np.array(preprocessed_data['Filter_size'])
    yp = np.array(preprocessed_data['nondimensional_ycoor'])
    xcoor = np.array(preprocessed_data['xcoor'])
    ycoor = np.array(preprocessed_data['ycoor'])
    zcoor = np.array(preprocessed_data['zcoor'])
    filtered_xcoor = np.array(preprocessed_data['filtered_xcoor'])
    filtered_ycoor = np.array(preprocessed_data['filtered_ycoor'])
    filtered_zcoor = np.array(preprocessed_data['filtered_zcoor'])
    preprocessed_data.close()
    
    #loop spatially through the xyz grid
    for i in range(len(ubox[0])):
        for j in range(len(ubox[0])):
            for k in range(len(ubox[0])):
                
                #add the filtered velocity components
                ubox_train[counter, :] = ubox[:, i, j, k]
                
                #add the filtered gradient components
                filtered_gradient[counter, :] = filtered_ugradient[:, :, i, j, k].flatten()
                
                #add the filtered strain tensor components (only 6 because symmetry)
                strain_tensor_train[counter, 0] = strain_tensor[0, 0, i, j, k]
                strain_tensor_train[counter, 1] = strain_tensor[1, 1, i, j, k]
                strain_tensor_train[counter, 2] = strain_tensor[2, 2, i, j, k]
                strain_tensor_train[counter, 3] = strain_tensor[0, 1, i, j, k]
                strain_tensor_train[counter, 4] = strain_tensor[0, 2, i, j, k]
                strain_tensor_train[counter, 5] = strain_tensor[1, 2, i, j, k]
                
                #add the residual stress tensor components (only 6 because symmetry)
                Output[counter, 0] = residual_stress[0, 0, i, j, k]
                Output[counter, 1] = residual_stress[1, 1, i, j, k]
                Output[counter, 2] = residual_stress[2, 2, i, j, k]
                Output[counter, 3] = residual_stress[0, 1, i, j, k]
                Output[counter, 4] = residual_stress[0, 2, i, j, k]
                Output[counter, 5] = residual_stress[1, 2, i, j, k]
                
#                SGS_frac[counter, 0] = frac
                
                Filter_size [counter, 0] = 1/filter_size
                
                yplus[counter, 0] = yp[j]
                
                counter += 1
    
    
    
    #scale resulting arrays
    ubox_train = (ubox_train - ubox_min) / ubox_scale
    filtered_gradient = (filtered_gradient - filtered_grad_min) / filtered_grad_scale
    strain_tensor_train = (strain_tensor_train - strain_tensor_min) / strain_tensor_scale
    yplus = (yplus - yplus_min) / yplus_scale
    Output = (Output - Output_min) / Output_scale
    
    #put the relevant input components into a single input array
    for i in range(len(filtered_gradient)):
        row = []
        for e in ubox_train[i, :]:
            row.append(e)
        for e in filtered_gradient[i, :]:
            row.append(e)
        for e in strain_tensor_train[i, :]:
            row.append(e)
#        row.append(SGS_frac[i, 0])
        row.append(Filter_size[i, 0])
        row.append(yplus[i, 0])
        Input.append(row)
    Input = np.array(Input)
    
    
    return Input, Output, datacube_size, filter_size, xcoor, ycoor, zcoor, filtered_xcoor, filtered_ycoor, filtered_zcoor, strain_tensor




epochs_plot = np.arange(1, n_epochs+1, 1)


fig1 = pl.figure()
subplot = fig1.add_subplot(111)
cc0_plot = subplot.plot(epochs_plot, history['cc_0'][0:n_epochs], color = 'red', label = r'$\tau_{11}$')
cc1_plot = subplot.plot(epochs_plot, history['cc_1'][0:n_epochs], color = 'blue', label = r'$\tau_{22}$')
cc2_plot = subplot.plot(epochs_plot, history['cc_2'][0:n_epochs], color = 'darkgreen', label = r'$\tau_{33}$')
cc3_plot = subplot.plot(epochs_plot, history['cc_3'][0:n_epochs], color = 'lime', label = r'$\tau_{12}$')
cc4_plot = subplot.plot(epochs_plot, history['cc_4'][0:n_epochs], color = 'cyan', label = r'$\tau_{13}$')
cc5_plot = subplot.plot(epochs_plot, history['cc_5'][0:n_epochs], color = 'magenta', label = r'$\tau_{23}$')
subplot.grid(True)
subplot.set_xlabel('Epochs [-]')
subplot.set_ylabel('Normalised training correlation coefficient [-]')
subplot.set_xlim((0, n_epochs))
subplot.set_ylim((0, 1))
subplot.legend()


fig2 = pl.figure()
subplot = fig2.add_subplot(111)
cc0_plot = subplot.plot(epochs_plot, history['val_cc_0'][0:n_epochs], color = 'red', label = r'$\tau_{11}$')
cc1_plot = subplot.plot(epochs_plot, history['val_cc_1'][0:n_epochs], color = 'blue', label = r'$\tau_{22}$')
cc2_plot = subplot.plot(epochs_plot, history['val_cc_2'][0:n_epochs], color = 'darkgreen', label = r'$\tau_{33}$')
cc3_plot = subplot.plot(epochs_plot, history['val_cc_3'][0:n_epochs], color = 'lime', label = r'$\tau_{12}$')
cc4_plot = subplot.plot(epochs_plot, history['val_cc_4'][0:n_epochs], color = 'cyan', label = r'$\tau_{13}$')
cc5_plot = subplot.plot(epochs_plot, history['val_cc_5'][0:n_epochs], color = 'magenta', label = r'$\tau_{23}$')
subplot.grid(True)
subplot.set_xlabel('Epochs [-]')
subplot.set_ylabel('Normalised validation correlation coefficient [-]')
subplot.set_xlim((0, n_epochs))
subplot.set_ylim((0, 1))
subplot.legend()



fig3 = pl.figure()
subplot = fig3.add_subplot(111)
loss_plot = subplot.plot(epochs_plot, history['loss'], color = 'blue', label = 'training')
val_loss_plot = subplot.plot(epochs_plot, history['val_loss'], color = 'red', label = 'validation')
subplot.grid(True)
subplot.set_xlabel('Epochs [-]')
subplot.set_ylabel('MSE [-]')
subplot.set_xlim((0, n_epochs))
subplot.set_ylim((0, 0.001))
subplot.legend()



train_cc = np.mean([history['cc_0'],
                    history['cc_1'],
                    history['cc_2'],
                    history['cc_3'],
                    history['cc_4'],
                    history['cc_5']],
                axis = 0)


val_cc = np.mean([history['val_cc_0'],
                  history['val_cc_1'],
                  history['val_cc_2'],
                  history['val_cc_3'],
                  history['val_cc_4'],
                  history['val_cc_5']],
                axis = 0)



fig4 = pl.figure()
subplot = fig4.add_subplot(111)
cc_train_plot = subplot.plot(epochs_plot, train_cc, color = 'red', label = 'training')
cc_val_plot = subplot.plot(epochs_plot, val_cc, color = 'blue', label = 'validation')
subplot.grid(True)
subplot.set_xlabel('Epochs [-]')
subplot.set_ylabel('Normalised Correlation coefficient [-]')
subplot.set_xlim((0, n_epochs))
subplot.set_ylim((0, 1))
subplot.legend()











if save:
    directory = str(NN_name) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
    #save all figures
    
    dir1 = directory + str('training_cc vs epochs')
    fig1.savefig(dir1)
    dir2 = directory + str('validation_cc vs epochs')
    fig2.savefig(dir2)
    dir3 = directory + str('MSE vs epochs')
    fig3.savefig(dir3)
    dir4 = directory + str('mean_cc vs epochs')
    fig4.savefig(dir4)








#lists for errors in \taut_ij for different filter sizes

re_max_11 = []
re_max_22 = []
re_max_33 = []
re_max_12 = []
re_max_23 = []
re_max_13 = []

re_mean_11 = []
re_mean_22 = []
re_mean_33 = []
re_mean_12 = []
re_mean_13 = []
re_mean_23 = []


FS = []



for test_file in test_files:


    test_input, test_output, datacube_size, filter_size, xcoor, ycoor, zcoor, filtered_xcoor, filtered_ycoor, filtered_zcoor, strain_tensor = shape_test_data(test_file,
                                                                                                                                               ubox_scale,
                                                                                                                                               filtered_grad_scale,
                                                                                                                                               strain_tensor_scale,
                                                                                                                                               yplus_scale,
                                                                                                                                               Output_scale,
                                                                                                                                               ubox_min,
                                                                                                                                               filtered_grad_min,
                                                                                                                                               strain_tensor_min,
                                                                                                                                               yplus_min,
                                                                                                                                               Output_min)
    
    prediction = model.predict(test_input)
    prediction = prediction * Output_scale + Output_min
    
    test_output = test_output * Output_scale + Output_min
    
    
    
    
    # error calculations
    FS.append(filter_size)
    
    re_max_11.append(np.amax(np.abs((test_output[:, 0] - prediction[:, 0]) / 1.2**2)))
    re_max_22.append(np.amax(np.abs((test_output[:, 1] - prediction[:, 1]) / 1.2**2)))
    re_max_33.append(np.amax(np.abs((test_output[:, 2] - prediction[:, 2]) / 1.2**2)))
    re_max_12.append(np.amax(np.abs((test_output[:, 3] - prediction[:, 3]) / 1.2**2)))
    re_max_23.append(np.amax(np.abs((test_output[:, 4] - prediction[:, 4]) / 1.2**2)))
    re_max_13.append(np.amax(np.abs((test_output[:, 5] - prediction[:, 5]) / 1.2**2)))
    
    re_mean_11.append(np.mean(np.abs((test_output[:, 0] - prediction[:, 0]) / 1.2**2)))
    re_mean_22.append(np.mean(np.abs((test_output[:, 1] - prediction[:, 1]) / 1.2**2)))
    re_mean_33.append(np.mean(np.abs((test_output[:, 2] - prediction[:, 2]) / 1.2**2)))
    re_mean_12.append(np.mean(np.abs((test_output[:, 3] - prediction[:, 3]) / 1.2**2)))
    re_mean_13.append(np.mean(np.abs((test_output[:, 4] - prediction[:, 4]) / 1.2**2)))
    re_mean_23.append(np.mean(np.abs((test_output[:, 5] - prediction[:, 5]) / 1.2**2)))
    
    
    
    
    
    
    
    pointer = 0
    predicted_stress = np.ndarray(shape = (3, 3, datacube_size, datacube_size, datacube_size))
    true_stress = np.ndarray(shape = (3, 3, datacube_size, datacube_size, datacube_size))
    smag_stress = np.ndarray(shape = (3, 3, datacube_size, datacube_size, datacube_size))
    Delta_x = np.ndarray(shape = (datacube_size))
    Delta_y = np.ndarray(shape = (datacube_size))
    Delta_z = np.ndarray(shape = (datacube_size))
    
    for k in range(0, datacube_size):
        if k == 0:
            Delta_z[k] = filtered_zcoor[k+1] / 2 - zcoor[0]
        elif k == datacube_size - 1:
            Delta_z[k] = zcoor[-1] - filtered_zcoor[k-1] / 2
        else:
            Delta_z[k] = (filtered_zcoor[k+1] - filtered_zcoor[k-1]) / 2
        
        for j in range(0, datacube_size):
            if j == 0:
                Delta_y[j] = filtered_ycoor[j+1] / 2 - ycoor[0]
            elif j == datacube_size - 1:
                Delta_y[j] = ycoor[-1] - filtered_ycoor[j-1] / 2
            else:
                Delta_y[j] = (filtered_ycoor[j+1] - filtered_ycoor[j-1]) / 2
            
            for i in range(0, datacube_size):
                predicted_stress[0, 0, k, j, i] = prediction[pointer, 0]
                predicted_stress[1, 1, k, j, i] = prediction[pointer, 1]
                predicted_stress[2, 2, k, j, i] = prediction[pointer, 2]
                predicted_stress[0, 1, k, j, i] = prediction[pointer, 3]
                predicted_stress[1, 0, k, j, i] = prediction[pointer, 3]
                predicted_stress[0, 2, k, j, i] = prediction[pointer, 4]
                predicted_stress[2, 0, k, j, i] = prediction[pointer, 4]
                predicted_stress[1, 2, k, j, i] = prediction[pointer, 5]
                predicted_stress[2, 1, k, j, i] = prediction[pointer, 5]
                
                true_stress[0, 0, k, j, i] = test_output[pointer, 0]
                true_stress[1, 1, k, j, i] = test_output[pointer, 1]
                true_stress[2, 2, k, j, i] = test_output[pointer, 2]
                true_stress[0, 1, k, j, i] = test_output[pointer, 3]
                true_stress[1, 0, k, j, i] = test_output[pointer, 3]
                true_stress[0, 2, k, j, i] = test_output[pointer, 4]
                true_stress[2, 0, k, j, i] = test_output[pointer, 4]
                true_stress[1, 2, k, j, i] = test_output[pointer, 5]
                true_stress[2, 1, k, j, i] = test_output[pointer, 5]
                pointer += 1
                
                if i == 0:
                    Delta_x[i] = filtered_xcoor[i+1] / 2 - xcoor[0]
                elif i == datacube_size - 1:
                    Delta_x[i] = xcoor[-1] - filtered_xcoor[i-1] / 2
                else:
                    Delta_x[i] = (filtered_xcoor[i+1] - filtered_xcoor[i-1]) / 2
                
                
                nu_t = (Delta_x[i] * Delta_y[j] * Delta_z[k])**(2/3) * C * np.sqrt(2 * np.tensordot(strain_tensor[:, :, k, j, i], strain_tensor[:, :, k, j, i], axes=2))
                for m in range(3):
                    for n in range(3):
                        if m == n:
                            smag_stress[m, n, k, j, i] = -3 * nu_t * strain_tensor[m, n, k, j, i]
                        else:
                            smag_stress[m, n, k, j, i] = -2 * nu_t * strain_tensor[m, n, k, j, i]
    
    
    
    
    
    pre_reshaping_predicted_stress = np.copy(predicted_stress)
    predicted_stress = np.ndarray((3, 3, image_resolution, image_resolution, image_resolution))
    
    pre_reshaping_true_stress = np.copy(true_stress)
    true_stress = np.ndarray((3, 3, image_resolution, image_resolution, image_resolution))
    
    pre_reshaping_smag_stress = np.copy(smag_stress)
    smag_stress = np.ndarray((3, 3, image_resolution, image_resolution, image_resolution))
    
    
    for i in range(3):
        for j in range(3):
            zoom = image_resolution / datacube_size
            predicted_stress[i, j, :, :, :] = scipy.ndimage.zoom(pre_reshaping_predicted_stress[i, j, :, :, :], zoom)
            true_stress[i, j, :, :, :] = scipy.ndimage.zoom(pre_reshaping_true_stress[i, j, :, :, :], zoom)
            smag_stress[i, j, :, :, :] = scipy.ndimage.zoom(pre_reshaping_smag_stress[i, j, :, :, :], zoom)
    
    
    
    
    
    if save:
        directory = str(NN_name) + '/' + test_file + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    
    
    
    
    x, y, z = 2, 2, 2
    
    
    for j in range(3):
        fig5 = pl.figure(figsize = (12, 12))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            subplot1 = fig5.add_subplot(3, 3, counter)
            a = subplot1.imshow(true_stress[j, i, z, :, :], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, z, :, :].min(), predicted_stress[j, i, z, :, :].min()), vmax = max(true_stress[j, i, z, :, :].max(), predicted_stress[j, i, z, :, :].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot1.title.set_text('True $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for z = ' + str(z))
            subplot2 = fig5.add_subplot(3, 3, counter+3)
            a = subplot2.imshow(predicted_stress[j, i, z, :, :], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, z, :, :].min(), predicted_stress[j, i, z, :, :].min()), vmax = max(true_stress[j, i, z, :, :].max(), predicted_stress[j, i, z, :, :].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot2.title.set_text('ANN predicted $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for z = ' + str(z))
            subplot3 = fig5.add_subplot(3, 3, counter+6)
            a = subplot3.imshow(smag_stress[j, i, z, :, :], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, z, :, :].min(), predicted_stress[j, i, z, :, :].min()), vmax = max(true_stress[j, i, z, :, :].max(), predicted_stress[j, i, z, :, :].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot3.title.set_text('Smagorinsky $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for z = ' + str(z))
            
            counter += 1
        
        if save:
            dir5 = directory + str('true vs predicted u_'+str(j+1)+'u_i for z = ' + str(z))
            fig5.savefig(dir5)
    
    
    for j in range(3):
        fig5 = pl.figure(figsize = (12, 12))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            subplot1 = fig5.add_subplot(3, 3, counter)
            a = subplot1.imshow(true_stress[j, i, :, y, :], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, :, y, :].min(), predicted_stress[j, i, :, y, :].min()), vmax = max(true_stress[j, i, :, y, :].max(), predicted_stress[j, i, :, y, :].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot1.title.set_text('True $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for y = ' + str(y))
            subplot2 = fig5.add_subplot(3, 3, counter+3)
            a = subplot2.imshow(predicted_stress[j, i, :, y, :], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, :, y, :].min(), predicted_stress[j, i, :, y, :].min()), vmax = max(true_stress[j, i, :, y, :].max(), predicted_stress[j, i, :, y, :].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot2.title.set_text('ANN predicted $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for y = ' + str(y))
            subplot3 = fig5.add_subplot(3, 3, counter+6)
            a = subplot3.imshow(smag_stress[j, i, :, y, :], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, :, y, :].min(), predicted_stress[j, i, :, y, :].min()), vmax = max(true_stress[j, i, :, y, :].max(), predicted_stress[j, i, :, y, :].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot3.title.set_text('Smagorinsky $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for y = ' + str(y))
            
            counter += 1
            
        if save:
            dir5 = directory + str('true vs predicted u_'+str(j+1)+'u_i for y = ' + str(z))
            fig5.savefig(dir5)
    
    
    for j in range(3):
        fig5 = pl.figure(figsize = (12, 12))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            subplot1 = fig5.add_subplot(3, 3, counter)
            a = subplot1.imshow(true_stress[j, i, :, :, x], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, :, :, x].min(), predicted_stress[j, i, :, :, x].min()), vmax = max(true_stress[j, i, :, :, x].max(), predicted_stress[j, i, :, :, x].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot1.title.set_text('True $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for x = ' + str(x))
            subplot2 = fig5.add_subplot(3, 3, counter+3)
            a = subplot2.imshow(predicted_stress[j, i, :, :, x], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, :, :, x].min(), predicted_stress[j, i, :, :, x].min()), vmax = max(true_stress[j, i, :, :, x].max(), predicted_stress[j, i, :, :, x].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot2.title.set_text('ANN predicted $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for x = ' + str(x))
            subplot3 = fig5.add_subplot(3, 3, counter+6)
            a = subplot3.imshow(smag_stress[j, i, :, :, x], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, :, :, x].min(), predicted_stress[j, i, :, :, x].min()), vmax = max(true_stress[j, i, :, :, x].max(), predicted_stress[j, i, :, :, x].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot3.title.set_text('Smagorinsky $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for x = ' + str(x))
            
            counter += 1
            
        if save:
            dir5 = directory + str('true vs predicted u_'+str(j+1)+'u_i for x = ' + str(z))
            fig5.savefig(dir5)

    



filter_size = FS

fig6 = pl.figure(figsize = (12, 5))
pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)

subplot6a = fig6.add_subplot(121)
subplot6a.scatter(filter_size, re_max_11, marker = '^', color = 'red', label = r'$\tau_{11}$')
subplot6a.scatter(filter_size, re_max_22, marker = '>', color = 'blue', label = r'$\tau_{22}$')
subplot6a.scatter(filter_size, re_max_33, marker = '<', color = 'darkgreen', label = r'$\tau_{33}$')
subplot6a.scatter(filter_size, re_max_12, marker = 'x', color = 'lime', label = r'$\tau_{12}$')
subplot6a.scatter(filter_size, re_max_13, marker = 'o', color = 'cyan', label = r'$\tau_{13}$')
subplot6a.scatter(filter_size, re_max_23, marker = 'v', color = 'magenta', label = r'$\tau_{23}$')
subplot6a.title.set_text('Max RE vs filter size')
subplot6a.set_xlabel('Filter size [-]')
subplot6a.set_ylabel('Maximum Relative Error [-]')
subplot6a.grid(True)
subplot6a.legend()

subplot6a = fig6.add_subplot(122)
subplot6a.scatter(filter_size, re_mean_11, marker = '^', color = 'red', label = r'$\tau_{11}$')
subplot6a.scatter(filter_size, re_mean_22, marker = '>', color = 'blue', label = r'$\tau_{22}$')
subplot6a.scatter(filter_size, re_mean_33, marker = '<', color = 'darkgreen', label = r'$\tau_{33}$')
subplot6a.scatter(filter_size, re_mean_12, marker = 'x', color = 'lime', label = r'$\tau_{12}$')
subplot6a.scatter(filter_size, re_mean_13, marker = 'o', color = 'cyan', label = r'$\tau_{13}$')
subplot6a.scatter(filter_size, re_mean_23, marker = 'v', color = 'magenta', label = r'$\tau_{23}$')
subplot6a.title.set_text('MRE vs filter size')
subplot6a.set_xlabel('Filter size [-]')
subplot6a.set_ylabel('Mean Relative Error [-]')
subplot6a.grid(True)
subplot6a.legend()


if save:
    directory = str(NN_name) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    dir6 = directory + 'maxRE and MRE vs filter sizes'
    fig6.savefig(dir6)