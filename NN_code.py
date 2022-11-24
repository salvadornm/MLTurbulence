# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 11:53:53 2022

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




save = True
NN_name = 'NN_31'

filetype = 'isotropic1024coarse, isotropic4096, isotropic8192'
train_files = ['isotropic1024coarse_6b_processed.h5',
               'isotropic1024coarse_7b_processed.h5',
               'isotropic1024coarse_8b_processed.h5',
               'isotropic1024coarse_1g_processed.h5',
               'isotropic1024coarse_2g_processed.h5',
               'isotropic1024coarse_3g_processed.h5',
               'isotropic1024coarse_4g_processed.h5',
               'isotropic1024coarse_5g_processed.h5',
               'isotropic4096_6b_processed.h5',
               'isotropic4096_7b_processed.h5',
               'isotropic4096_8b_processed.h5',
               'isotropic4096_1g_processed.h5',
               'isotropic4096_2g_processed.h5',
               'isotropic4096_3g_processed.h5',
               'isotropic4096_4g_processed.h5',
               'isotropic4096_5g_processed.h5',
               'isotropic8192_6b_processed.h5',
               'isotropic8192_7b_processed.h5',
               'isotropic8192_8b_processed.h5',
               'isotropic8192_1g_processed.h5',
               'isotropic8192_2g_processed.h5',
               'isotropic8192_3g_processed.h5',
               'isotropic8192_4g_processed.h5',
               'isotropic8192_5g_processed.h5']

test_files = ['isotropic1024coarse_10b_processed.h5',
              'isotropic4096_10b_processed.h5',
              'isotropic8192_10b_processed.h5']
test_split = 0.2

n_epochs = 2
activation_function = 'tanh'
layer_MLPs = [24, 12, 24, 6]
loss_function = 'mse'
learning_rate = 0.001

image_resolution = 31




def shape_train_data (file):
    
    """
    Read the data from the hdf5 files and divide into input and output arrays.
    These contain information corresponding to a single spatial data point.
    The function takes the type of flow analysed and the number of data cubes.
    """
    
    array_size = 0
    
    #extract datacube size
    for number in range(0, len(file)):
        filename = file[number]
        preprocessed_data = h5py.File(filename, 'r')
        ubox = np.array(preprocessed_data['Velocity_Box_filter'])
    
        array_size += (len(ubox[0]))**3
    
    #create arrays for the single data types to be extracted
    ubox_train = np.ndarray((array_size, 3))
    filtered_gradient = np.ndarray((array_size, 9))
    strain_tensor_train = np.ndarray((array_size, 6))
    SGS_frac = np.ndarray((array_size, 1))
    Filter_size = np.ndarray((array_size, 1))
    Input = []
    Output = np.ndarray((array_size, 6))
    
    
    counter = 0
    
    
    #loop through all data cubes
    for number in range(0, len(file)):
        
        #read the file
        filename = file[number]
        preprocessed_data = h5py.File(filename, 'r')
        ubox = np.array(preprocessed_data['Velocity_Box_filter'])
        filtered_ugradient = np.array(preprocessed_data['Gradient_of_filtered_velocity'])
        strain_tensor = np.array(preprocessed_data['Strain_tensor'])
        residual_stress = np.array(preprocessed_data['Residual_stress'])
        frac = np.array(preprocessed_data['Fraction_of_subgrid_scale_energy'])
        filter_size = np.array(preprocessed_data['Fraction_of_subgrid_scale_energy'])
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
                    
                    SGS_frac[counter, 0] = frac
                    
                    Filter_size [counter, 0] = 1/filter_size
                    
                    counter += 1
        
        
        print('iteration ' + str(number+1) + ' of ' + str(len(file)))
    
    #find max and min of filtered velocity and scale based on their difference
    ubox_max = np.amax(ubox_train, axis = 0)
    ubox_min = np.amin(ubox_train, axis = 0)
    ubox_scale = ubox_max - ubox_min
    ubox_train = (ubox_train - ubox_min) / ubox_scale
    
    #find max and min of filtered velocity gradient and scale based on their difference
    filtered_grad_max = np.amax(filtered_gradient, axis = 0)
    filtered_grad_min = np.amin(filtered_gradient, axis = 0)
    filtered_grad_scale = filtered_grad_max - filtered_grad_min
    filtered_gradient = (filtered_gradient - filtered_grad_min) / filtered_grad_scale
    
    #find max and min of filtered strain tensor and scale based on their difference
    strain_tensor_max = np.amax(strain_tensor_train, axis = 0)
    strain_tensor_min = np.amin(strain_tensor_train, axis = 0)
    strain_tensor_scale = strain_tensor_max - strain_tensor_min
    strain_tensor_train = (strain_tensor_train - strain_tensor_min) / strain_tensor_scale
    
    #find max and min of residual stress tensor and scale based on their difference
    Output_max = np.amax(Output, axis = 0)
    Output_min = np.amin(Output, axis = 0)
    Output_scale = Output_max - Output_min
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
        row.append(SGS_frac[i, 0])
        row.append(Filter_size[i, 0])
        Input.append(row)
    Input = np.array(Input)
    
    
    return Input, Output, ubox_scale, filtered_grad_scale, strain_tensor_scale, Output_scale, ubox_min, filtered_grad_min, strain_tensor_min, Output_min








def shape_test_data (file, ubox_scale, filtered_grad_scale, strain_tensor_scale, Output_scale, ubox_min, filtered_grad_min, strain_tensor_min, Output_min):
    
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
    SGS_frac = np.ndarray((array_size, 1))
    Filter_size = np.ndarray((array_size, 1))
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
    filter_size = np.array(preprocessed_data['Fraction_of_subgrid_scale_energy'])
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
                
                SGS_frac[counter, 0] = frac
                
                Filter_size [counter, 0] = 1/filter_size
                
                counter += 1
        
        
    
    #scale resulting arrays
    ubox_train = (ubox_train - ubox_min) / ubox_scale
    filtered_gradient = (filtered_gradient - filtered_grad_min) / filtered_grad_scale
    strain_tensor_train = (strain_tensor_train - strain_tensor_min) / strain_tensor_scale
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
        row.append(SGS_frac[i, 0])
        row.append(Filter_size[i, 0])
        Input.append(row)
    Input = np.array(Input)
    
    
    return Input, Output, datacube_size








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















Input, Output, ubox_scale, filtered_grad_scale, strain_tensor_scale, Output_scale, ubox_min, filtered_grad_min, strain_tensor_min, Output_min = shape_train_data (train_files)
train_input, val_input, train_output, val_output = train_test_split(Input, Output, test_size = test_split)





keras.backend.clear_session()

#create the ML model by stacking layers
model = tf.keras.Sequential([
    keras.layers.Dense(24, activation = activation_function, input_dim = 20),
    keras.layers.Dense(12, activation = activation_function),
    keras.layers.Dense(24, activation = activation_function),
    keras.layers.Dense(6)
])


model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=loss_function, 
    metrics = [cc_0, cc_1, cc_2, cc_3, cc_4, cc_5])
start = timer()
history = model.fit(train_input, train_output, validation_data = (val_input, val_output), epochs=n_epochs, verbose = 1)
end = timer()

print(timedelta(seconds=end-start))



















epochs_plot = np.arange(1, n_epochs+1, 1)


fig1 = pl.figure()
subplot = fig1.add_subplot(111)
cc0_plot = subplot.plot(epochs_plot, history.history['cc_0'][0:n_epochs], color = 'red', label = r'$\tau_{11}$')
cc1_plot = subplot.plot(epochs_plot, history.history['cc_1'][0:n_epochs], color = 'blue', label = r'$\tau_{22}$')
cc2_plot = subplot.plot(epochs_plot, history.history['cc_2'][0:n_epochs], color = 'darkgreen', label = r'$\tau_{33}$')
cc3_plot = subplot.plot(epochs_plot, history.history['cc_3'][0:n_epochs], color = 'lime', label = r'$\tau_{12}$')
cc4_plot = subplot.plot(epochs_plot, history.history['cc_4'][0:n_epochs], color = 'cyan', label = r'$\tau_{13}$')
cc5_plot = subplot.plot(epochs_plot, history.history['cc_5'][0:n_epochs], color = 'magenta', label = r'$\tau_{23}$')
subplot.grid(True)
subplot.set_xlabel('Epochs [-]')
subplot.set_ylabel('Normalised training correlation coefficient [-]')
subplot.set_xlim((0, n_epochs))
subplot.set_ylim((0, 1))
subplot.legend()


fig2 = pl.figure()
subplot = fig2.add_subplot(111)
cc0_plot = subplot.plot(epochs_plot, history.history['val_cc_0'][0:n_epochs], color = 'red', label = r'$\tau_{11}$')
cc1_plot = subplot.plot(epochs_plot, history.history['val_cc_1'][0:n_epochs], color = 'blue', label = r'$\tau_{22}$')
cc2_plot = subplot.plot(epochs_plot, history.history['val_cc_2'][0:n_epochs], color = 'darkgreen', label = r'$\tau_{33}$')
cc3_plot = subplot.plot(epochs_plot, history.history['val_cc_3'][0:n_epochs], color = 'lime', label = r'$\tau_{12}$')
cc4_plot = subplot.plot(epochs_plot, history.history['val_cc_4'][0:n_epochs], color = 'cyan', label = r'$\tau_{13}$')
cc5_plot = subplot.plot(epochs_plot, history.history['val_cc_5'][0:n_epochs], color = 'magenta', label = r'$\tau_{23}$')
subplot.grid(True)
subplot.set_xlabel('Epochs [-]')
subplot.set_ylabel('Normalised validation correlation coefficient [-]')
subplot.set_xlim((0, n_epochs))
subplot.set_ylim((0, 1))
subplot.legend()



fig3 = pl.figure()
subplot = fig3.add_subplot(111)
loss_plot = subplot.plot(epochs_plot, history.history['loss'], color = 'blue', label = 'training')
val_loss_plot = subplot.plot(epochs_plot, history.history['val_loss'], color = 'red', label = 'validation')
subplot.grid(True)
subplot.set_xlabel('Epochs [-]')
subplot.set_ylabel('MSE [-]')
subplot.set_xlim((0, n_epochs))
subplot.set_ylim((0, 0.001))
subplot.legend()



train_cc = np.mean([history.history['cc_0'],
                    history.history['cc_1'],
                    history.history['cc_2'],
                    history.history['cc_3'],
                    history.history['cc_4'],
                    history.history['cc_5']],
                axis = 0)


val_cc = np.mean([history.history['val_cc_0'],
                  history.history['val_cc_1'],
                  history.history['val_cc_2'],
                  history.history['val_cc_3'],
                  history.history['val_cc_4'],
                  history.history['val_cc_5']],
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
    
    
    #save text with network information for reproducibility
    text_filename = directory + 'description.txt'
    MLP = str(layer_MLPs[0])
    for i in range(1, len(layer_MLPs)):
        MLP = MLP + ' x ' + str(layer_MLPs[i])
    MLP = MLP + '\n\n'
    
    with open(text_filename, 'w') as f:
        f.write('%%%%%%%%%%%%%%%%%%%%%%% ' + str(NN_name) + ' %%%%%%%%%%%%%%%%%%%%%%%\n\n')
        f.write('MLP structure:\n')
        f.write(MLP)
        f.write('n_epochs = ' + str(n_epochs) + '\n')
        f.write('activation_function = ' + str(activation_function) + '\n')
        f.write('input_type = <u>, <grad(u)>, <S>, SGS_fraction, filter_size\n')
        f.write('output_type = tau\n')
        f.write('training_size(cubes) = ' + str(len(train_files)) + '\n')
        f.write('train_test_split = ' + str(test_split) + '\n')
        f.write('loss = ' + str(loss_function) + '\n')
        f.write('optimiser = Adam\n')
        f.write('learning rate = ' + str(learning_rate) + '\n')
        f.write('cc_type = pearson' + '\n')
        f.write('training time = ' + str(timedelta(seconds = end - start)) + '\n')
    
    #save all figures
    
    dir1 = directory + str('training_cc vs epochs')
    fig1.savefig(dir1)
    dir2 = directory + str('validation_cc vs epochs')
    fig2.savefig(dir2)
    dir3 = directory + str('MSE vs epochs')
    fig3.savefig(dir3)
    dir4 = directory + str('mean_cc vs epochs')
    fig4.savefig(dir4)
    
    #save model
    model_directory = directory + str('model.h5')
    model.save(model_directory)
    
    #save model scaling parameters
    
    filename = directory + 'scaling_parameters.h5'
    scaling_params = h5py.File(filename, 'w')
    scaling_params.create_dataset('ubox_scale', data = ubox_scale)
    scaling_params.create_dataset('ubox_min', data = ubox_min)
    scaling_params.create_dataset('filtered_gradient_scale', data = filtered_grad_scale)
    scaling_params.create_dataset('filtered_gradient_min', data = filtered_grad_min)
    scaling_params.create_dataset('strain_tensor_scale', data = strain_tensor_scale)
    scaling_params.create_dataset('strain_tensor_min', data = strain_tensor_min)
    scaling_params.create_dataset('output_scale', data = Output_scale)
    scaling_params.create_dataset('output_min', data = Output_min)
    scaling_params.close()















for test_file in test_files:


    test_input, test_output, datacube_size = shape_test_data(test_file, ubox_scale, filtered_grad_scale, strain_tensor_scale, Output_scale, ubox_min, filtered_grad_min, strain_tensor_min, Output_min)
    
    prediction = model.predict(test_input)
    prediction = prediction * Output_scale + Output_min
    
    test_output = test_output * Output_scale + Output_min
    
    
    
    
    
    
    pointer = 0
    predicted_stress = np.zeros(shape = (3, 3, datacube_size, datacube_size, datacube_size))
    true_stress = np.zeros(shape = (3, 3, datacube_size, datacube_size, datacube_size))
    
    for k in range(0, datacube_size):
        for j in range(0, datacube_size):
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
    
    
    
    
    
    pre_reshaping_predicted_stress = np.copy(predicted_stress)
    predicted_stress = np.ndarray((3, 3, image_resolution, image_resolution, image_resolution))
    
    pre_reshaping_true_stress = np.copy(true_stress)
    true_stress = np.ndarray((3, 3, image_resolution, image_resolution, image_resolution))
    
    
    for i in range(3):
        for j in range(3):
            zoom = image_resolution / datacube_size
            predicted_stress[i, j, :, :, :] = scipy.ndimage.zoom(pre_reshaping_predicted_stress[i, j, :, :, :], zoom)
            true_stress[i, j, :, :, :] = scipy.ndimage.zoom(pre_reshaping_true_stress[i, j, :, :, :], zoom)
    
    
    
    
    
    if save:
        directory = str(NN_name) + '/' + test_file + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    
    
    
    
    x, y, z = 10, 10, 10
    
    
    for j in range(3):
        fig5 = pl.figure(figsize = (12, 6))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            subplot1 = fig5.add_subplot(2, 3, counter)
            a = subplot1.imshow(true_stress[j, i, z, :, :], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, z, :, :].min(), predicted_stress[j, i, z, :, :].min()), vmax = max(true_stress[j, i, z, :, :].max(), predicted_stress[j, i, z, :, :].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot1.title.set_text('true $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for z = ' + str(z))
            subplot2 = fig5.add_subplot(2, 3, counter+3)
            a = subplot2.imshow(predicted_stress[j, i, z, :, :], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, z, :, :].min(), predicted_stress[j, i, z, :, :].min()), vmax = max(true_stress[j, i, z, :, :].max(), predicted_stress[j, i, z, :, :].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot2.title.set_text('predicted $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for z = ' + str(z))
            
            counter += 1
        
        if save:
            dir5 = directory + str('true vs predicted u_'+str(j+1)+'u_i for z = ' + str(z))
            fig5.savefig(dir5)
    
    
    for j in range(3):
        fig5 = pl.figure(figsize = (12, 6))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            subplot1 = fig5.add_subplot(2, 3, counter)
            a = subplot1.imshow(true_stress[j, i, :, y, :], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, :, y, :].min(), predicted_stress[j, i, :, y, :].min()), vmax = max(true_stress[j, i, :, y, :].max(), predicted_stress[j, i, :, y, :].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot1.title.set_text('true $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for y = ' + str(y))
            subplot2 = fig5.add_subplot(2, 3, counter+3)
            a = subplot2.imshow(predicted_stress[j, i, :, y, :], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, :, y, :].min(), predicted_stress[j, i, :, y, :].min()), vmax = max(true_stress[j, i, :, y, :].max(), predicted_stress[j, i, :, y, :].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot2.title.set_text('predicted $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for y = ' + str(y))
            
            counter += 1
            
        if save:
            dir5 = directory + str('true vs predicted u_'+str(j+1)+'u_i for y = ' + str(z))
            fig5.savefig(dir5)
    
    
    for j in range(3):
        fig5 = pl.figure(figsize = (12, 6))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            subplot1 = fig5.add_subplot(2, 3, counter)
            a = subplot1.imshow(true_stress[j, i, :, :, x], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, :, :, x].min(), predicted_stress[j, i, :, :, x].min()), vmax = max(true_stress[j, i, :, :, x].max(), predicted_stress[j, i, :, :, x].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot1.title.set_text('true $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for x = ' + str(x))
            subplot2 = fig5.add_subplot(2, 3, counter+3)
            a = subplot2.imshow(predicted_stress[j, i, :, :, x], interpolation = 'none', cmap = cm.jet, vmin = min(true_stress[j, i, :, :, x].min(), predicted_stress[j, i, :, :, x].min()), vmax = max(true_stress[j, i, :, :, x].max(), predicted_stress[j, i, :, :, x].max()))
            pl.colorbar(a, fraction = 0.045)
            subplot2.title.set_text('predicted $\overline{u\'_'+str(j+1)+'u\'_'+str(i+1)+'}$ for x = ' + str(x))
            
            counter += 1
            
        if save:
            dir5 = directory + str('true vs predicted u_'+str(j+1)+'u_i for x = ' + str(z))
            fig5.savefig(dir5)





