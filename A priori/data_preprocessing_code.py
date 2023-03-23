# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:34:50 2022

@author: P27
"""

#import all necessary libraries
import numpy as np
import h5py
import matplotlib.pyplot as pl
from matplotlib import cm
import random as rn
import scipy.ndimage
from scipy import signal

for dataset_number in range(21, 31):
    
    
#extract the raw data from the hdf5 files
    file = h5py.File('channel5200_' + str(dataset_number) + '.h5', 'r')
    filep = h5py.File('channel5200_' + str(dataset_number) + '_pressure.h5', 'r')
    keys = list(file.keys())
    key = list(filep.keys())
    print(keys)
    print(key)
    u = np.array(file[keys[0]])
    p = np.array(filep[key[0]])
    xcoor = np.array(file['xcoor'])
    ycoor = np.array(file['ycoor'])
    zcoor = np.array(file['zcoor'])
    file.close()
    filep.close()
    
    ycoor_min = np.amin(ycoor)
    ycoor = ycoor - ycoor_min
        
    
    plot = True
    
    
    #filename = 'data1_processed'
    #preprocessed_data = h5py.File(filename, 'r')
    #u = np.array(preprocessed_data['Velocity_0001'])
    #p = np.array(preprocessed_data['Pressure_0001'])
    #xcoor = np.array(preprocessed_data['xcoor'])
    #ycoor = np.array(preprocessed_data['ycoor'])
    #zcoor = np.array(preprocessed_data['zcoor'])
    #ugauss = np.array(preprocessed_data['Velocity_Gaussian_filter'])
    #ubox = np.array(preprocessed_data['Velocity_Box_filter'])
    #filtered_ugradient = np.array(preprocessed_data['Gradient_of_filtered_velocity'])
    #strain_tensor = np.array(preprocessed_data['Strain_tensor'])
    #residual_stress = np.array(preprocessed_data['Residual_stress'])
    #residual_stress_gradient = np.array(preprocessed_data['Residual_stress_gradient'])
    #preprocessed_data.close()
    
    
    
    
    
    
    
    """
    
    ---------------------------------------------------
    
    DEFINE ALL MAJOR FUNCTIONS FOR DATA PREPROCESSING
    
    ---------------------------------------------------
    
    """
    
    
    
    
    
    
    #evaluate the gradient by creating a function that receives the velocity tensor and the coordinates in x, y and z directions
    def gradu(u, xcoor, ycoor, zcoor):
        len_x = len(xcoor)
        len_y = len(ycoor)
        len_z = len(zcoor)
        #create a gradient list
        grad = np.ndarray((3, 3, len_x, len_y, len_z))
        #loop through all 3 velocity components
        for comp in range(0, 3):
            for k in range(0, len_z):
                for j in range(0, len_y):
                    grad[comp, 0, k, j, :] = np.gradient(u[k, j, :, comp], xcoor)
                for i in range(0, len_x):
                    grad[comp, 1, k, :, i] = np.gradient(u[k, :, i, comp], ycoor)
            for j in range(0, len_y):
                for i in range(0, len_x):
                    grad[comp, 2, :, j, i] = np.gradient(u[:, j, i, comp], zcoor)
        return grad
    
    
    
    
    def Box_filter3D(data, filter_size):
        """ This function calculates the filtered
            data through a box filter of specified size
        """
    
        def cube_matrix(cube):
            """ This function calculates the mean value
                of each data fed through the filter
            """
            #cube = np.array(cube)       #transrorm the square into a numpy array
            tot_sum = np.sum(cube)        #sum all elements in the array
                    
            return tot_sum / (filter_size**3)     # return the average of the sum of pixels
        
        cube = []
                          
        filtered_row = []   # Here we will store the resulting filtered
                            # data possible in one row 
                            # and will append this in the filtered_data
        filtered_square = []
        
        
        filtered_cube = [] # This is the resulting blurred image
        
        
        # number of squares in the given data
        n_squares = len(data) 
        
        # number of rows in the given data
        n_rows = len(data[0]) 
          
        # number of columns in the given data
        n_col = len(data[0, 0]) 
          
        # rp is row pointer and cp is column pointer
        sp, rp, cp, iteration = 0, 0, 0, 0
          
        # These while loops will be used to 
        # calculate all the filtered data in each row 
        while sp <= n_squares - filter_size:
            while rp <= n_rows - filter_size: 
                while cp <= n_col - filter_size:
                    
                    cube = data[sp : sp + filter_size, rp : rp + filter_size, cp : cp + filter_size]
                    
                    # calculate the filtered data for given n * n * n tensor
                    # i.e. cube and append it in filtered_row
                    filtered_row.append(cube_matrix(cube))
                    cube = []
                    
                    
                    # increase the column pointer
                    cp = cp + filter_size
                  
                # append the filtered_row in filtered_square
                filtered_square.append(filtered_row)
                filtered_row = []
                rp = rp + filter_size # increase row pointer
                cp = 0 # reset column pointer
                
            # append the filtered_square in filtered_cube
            filtered_cube.append(filtered_square)
            filtered_square = []
            sp = sp + filter_size # increase square pointer
            rp = 0 # reset row pointer
            
            iteration += 1
            print('iteration ' + str(iteration) + ' of ' + str(round(n_squares/filter_size)) + ' complete')
            
        # Reshape and return the resulting filtered data
        reshaped_filtered_data = np.array(filtered_cube)
        #zoom = n_rows / len(filtered_cube)
        #reshaped_filtered_data = scipy.ndimage.interpolation.zoom(filtered_cube, zoom)
        
        return reshaped_filtered_data
    
    
    
    
    
    def Gaussian_filter3D(data, filter_size):
        """ This function calculates the filtered
            data through a box filter of specified size
        """
        
        def cube_matrix(cube, gaussian_kernel_3D):
            """ This function calculates the Gaussian filtered value
                of each data fed through the filter
            """
            
            cube = np.array(cube)
            
            filtered_cube = cube * gaussian_kernel_3D       #apply the filter
                    
            return np.sum(filtered_cube) / np.sum(gaussian_kernel_3D)    # return the sum of the filtered components
        
        std = np.pi * filter_size**3 / (12 * np.sqrt(6))
        gaussian_kernel_1D = signal.gaussian(filter_size, std).reshape(filter_size, 1)
        gaussian_kernel_2D = np.outer(gaussian_kernel_1D, gaussian_kernel_1D)
        gaussian_kernel_product = np.outer(gaussian_kernel_2D, gaussian_kernel_1D)
        gaussian_kernel_3D = gaussian_kernel_product.reshape((filter_size, filter_size, filter_size))
    
        
    
        cube = []
        
                          
        filtered_row = []   # Here we will store the resulting filtered
                            # data possible in one row 
                            # and will append this in the filtered_data
        filtered_square = []
        
        
        filtered_cube = [] # This is the resulting blurred image
        
        
        # number of squares in the given data
        n_squares = len(data) 
        
        # number of rows in the given data
        n_rows = len(data[0]) 
          
        # number of columns in the given data
        n_col = len(data[0, 0]) 
          
        # rp is row pointer and cp is column pointer
        sp, rp, cp, iteration = 0, 0, 0, 0
          
        # These while loops will be used to 
        # calculate all the filtered data in each row 
        while sp <= n_squares - filter_size:        
            while rp <= n_rows - filter_size: 
                while cp <= n_col - filter_size:
                    
                    cube = data[sp : sp + filter_size, rp : rp + filter_size, cp : cp + filter_size]
                    
                    # calculate the filtered data for given n * n * n tensor
                    # i.e. cube and append it in filtered_row
                    filtered_row.append(cube_matrix(cube, gaussian_kernel_3D))
                    cube = []
                          
                    # increase the column pointer
                    cp = cp + filter_size
                  
                # append the filtered_row in filtered_square
                filtered_square.append(filtered_row)
                filtered_row = []
                rp = rp + filter_size # increase row pointer
                cp = 0 # reset column pointer
                
            # append the filtered_square in filtered_cube
            filtered_cube.append(filtered_square)
            filtered_square = []
            sp = sp + filter_size # increase square pointer
            rp = 0 # reset row pointer
            
            iteration += 1
            print('iteration ' + str(iteration) + ' of ' + str(round(n_squares/filter_size)) + ' complete')
          
        # Reshape and return the resulting filtered data
        reshaped_filtered_data = np.array(filtered_cube)
        #zoom = n_rows / len(filtered_cube)
        #reshaped_filtered_data = scipy.ndimage.interpolation.zoom(filtered_cube, zoom)
        
        return reshaped_filtered_data
    
    
    
    
    def filtered_coor (xcoor, ycoor, zcoor, filter_size):
        
        filtered_xcoor = []
        filtered_ycoor = []
        filtered_zcoor = []
        
        i = 0
        
        while i <= len(xcoor) - filter_size:
            filtered_xcoor.append(xcoor[i + filter_size//2])
            filtered_ycoor.append(ycoor[i + filter_size//2])
            filtered_zcoor.append(zcoor[i + filter_size//2])
            
            i += filter_size
        
        filtered_xcoor = np.array(filtered_xcoor)
        filtered_ycoor = np.array(filtered_ycoor)
        filtered_zcoor = np.array(filtered_zcoor)
        
        return filtered_xcoor, filtered_ycoor, filtered_zcoor
    
    
    
    
    
    
    def residual_stress (u, ufilter, Filter, filter_size):
        """ This function calculates the reynolds stress given the velocity vector,
            the filter function and the filter size
        """
        stress = []         #create a list for the stress tensor
        counter = 1
        #loop through all components of the tensor
        for i in range(3):
            row = []        #create a list for each row in the stress tensor
            for j in range(3):
                
                print('calculating stress component ' + str(counter) + ' of ' + str(9) + '\n')
                
                product = u[:, :, :, i] * u[:, :, :, j]                   #find the product between u_i and u_j for each spatial point
                filtered_product = Filter(product, filter_size)           #filter the product using a preselected filter
                #filtered_u_i = Filter(u[:, :, :, i], filter_size)
                #filtered_u_j = Filter(u[:, :, :, j], filter_size)
                filtered_u_i = ufilter[:, :, :, i]
                filtered_u_j = ufilter[:, :, :, j]
                stress_value = filtered_product - filtered_u_i * filtered_u_j
                row.append(stress_value)                                  #append the result to the row list
                
                print('\n stress component ' + str(counter) + ' of ' + str(9) + ' found \n -------------------------------- \n \n')
                counter += 1
            
            
            stress.append(row)                                  #append the row to the stress tensor list
        
        #return the resulting stress tensor as a numpy array
        return np.array(stress)
    
    
    
    
    #evaluate the gradient by creating a function that receives the velocity tensor and the coordinates in x, y and z directions
    def gradustress(ustress, xcoor, ycoor, zcoor):
        len_x = len(xcoor)
        len_y = len(ycoor)
        len_z = len(zcoor)
        #create a gradient list
        grad = np.ndarray((3, 3, 3, len_x, len_y, len_z))
        counter = 1
        #loop through all stress components
        for l in range(3):
            for m in range(3):
                
                print('calculating gradient of component ' + str(counter) + ' of 9 \n')
                
                for k in range(0, len_z):
                    for j in range(0, len_y):
                        grad[l, m, 0, k, j, :] = np.gradient(ustress[l, m, k, j, :], xcoor)
                    for i in range(0, len_x):
                        grad[l, m, 1, k, :, i] = np.gradient(ustress[l, m, k, :, i], ycoor)
                for j in range(0, len_y):
                    for i in range(0, len_x):
                        grad[l, m, 2, :, j, i] = np.gradient(ustress[l, m, :, j, i], zcoor)
                
                counter += 1
        
        return grad
    
    
    
    
    def FilteredStrainTensor (filtered_u_grad):
        """
        Define a function that given the gradient of the filtered velocity field finds the
        filtered strain tensor
        """
        S = []
        for i in range(3):
            row = []
            for j in range (3):
                S_ij = 1/2 * (filtered_u_grad[i, j, :, :, :] + filtered_u_grad[j, i, :, :, :])
                row.append(S_ij)
            S.append(row)
        return np.array(S)
    
    
    
    
    
    """
    
    --------------------------------------------
    
    TRANSPOSE DATA TO CREATE GALILEAN INVARIANCE
    
    --------------------------------------------
    
    """
    
#    u_original = np.copy(u)
#    p_original = np.copy(p)
#    
#    
#    for i in range(3):
#        u[:, :, :, i] = np.transpose(u_original[:, :, :, i], axes=(0,1,2))
#    p[:, :, :, 0] = np.transpose(p_original[:, :, :, 0], axes=(0,1,2))
    
    
    
    
    
    
    
    
    """
    
    --------------------------------------------
    
    APPLY THE ACTUAL PREPROCESSING
    
    --------------------------------------------
    
    """
    
    
    print('\n Preprocessing started. \n')
    
    #find the filtered velocity field
    
    filter_size = 8
    
    print('Performing Box filtering. \n')
    
    u1filtered_box = Box_filter3D(u[:, :, :, 0], filter_size)
    u2filtered_box = Box_filter3D(u[:, :, :, 1], filter_size)
    u3filtered_box = Box_filter3D(u[:, :, :, 2], filter_size)
    
    ubox = np.ndarray([len(u1filtered_box), len(u1filtered_box), len(u1filtered_box), 3])
    ubox[:, :, :, 0] = u1filtered_box
    ubox[:, :, :, 1] = u2filtered_box
    ubox[:, :, :, 2] = u3filtered_box
    
    print('\n Box filtering ended. \n Performing Gaussian filtering. \n')
    
    u1filtered_gauss = Gaussian_filter3D(u[:, :, :, 0], filter_size)
    u2filtered_gauss = Gaussian_filter3D(u[:, :, :, 1], filter_size)
    u3filtered_gauss = Gaussian_filter3D(u[:, :, :, 2], filter_size)
    
    ugauss = np.ndarray([len(u1filtered_gauss), len(u1filtered_gauss), len(u1filtered_gauss), 3])
    ugauss[:, :, :, 0] = u1filtered_gauss
    ugauss[:, :, :, 1] = u2filtered_gauss
    ugauss[:, :, :, 2] = u3filtered_gauss
    
    print('\n Gaussian filtering ended. \n Performing coordinate filtering. \n')
    
    filtered_xcoor, filtered_ycoor, filtered_zcoor = filtered_coor(xcoor, ycoor, zcoor, filter_size)
    
    
    
    #find the gradient and strain tensor of the filtered velocity field
    
    print('\n Finding the gradient of the filtered velocity field. \n')
    
    u_grad = gradu(u, xcoor, ycoor, zcoor)
    
#    filtered_ugradient_11 = Box_filter3D(u_grad[0, 0, :, :, :], filter_size)
#    filtered_ugradient_12 = Box_filter3D(u_grad[0, 1, :, :, :], filter_size)
#    filtered_ugradient_13 = Box_filter3D(u_grad[0, 2, :, :, :], filter_size)
#    filtered_ugradient_21 = Box_filter3D(u_grad[1, 0, :, :, :], filter_size)
#    filtered_ugradient_22 = Box_filter3D(u_grad[1, 1, :, :, :], filter_size)
#    filtered_ugradient_23 = Box_filter3D(u_grad[1, 2, :, :, :], filter_size)
#    filtered_ugradient_31 = Box_filter3D(u_grad[2, 0, :, :, :], filter_size)
#    filtered_ugradient_32 = Box_filter3D(u_grad[2, 1, :, :, :], filter_size)
#    filtered_ugradient_33 = Box_filter3D(u_grad[2, 2, :, :, :], filter_size)
#    
#    
#    filtered_ugradient = np.ndarray([3, 3, len(filtered_ugradient_11), len(filtered_ugradient_11), len(filtered_ugradient_11),])
#    
#    filtered_ugradient[0, 0, :, :, :] = filtered_ugradient_11
#    filtered_ugradient[0, 1, :, :, :] = filtered_ugradient_12
#    filtered_ugradient[0, 2, :, :, :] = filtered_ugradient_13
#    filtered_ugradient[1, 0, :, :, :] = filtered_ugradient_21
#    filtered_ugradient[1, 1, :, :, :] = filtered_ugradient_22
#    filtered_ugradient[1, 2, :, :, :] = filtered_ugradient_23
#    filtered_ugradient[2, 0, :, :, :] = filtered_ugradient_31
#    filtered_ugradient[2, 1, :, :, :] = filtered_ugradient_32
#    filtered_ugradient[2, 2, :, :, :] = filtered_ugradient_33
    
    
    
    
    filtered_ugradient = gradu(ubox, filtered_xcoor, filtered_ycoor, filtered_zcoor)
    
    print('\n Finding the strain tensor field. \n')
    
    strain_tensor = FilteredStrainTensor(filtered_ugradient)
    
    
    
    #find the residual stress field and gradient of the residual stress field
    
    print('Finding the residual stress tensor. \n')
    
    residual_stress = residual_stress(u, ubox, Box_filter3D, filter_size)
    
    print('\n Finding the gradient of the residual stress tensor. \n')
    
    residual_stress_gradient = gradustress(residual_stress, filtered_xcoor, filtered_ycoor, filtered_zcoor)
    
    
    
    print('\n \n Data preprocessing ended. \n')
    
    reshaped_ubox = np.ndarray((u.shape))
    
    for i in range(3):
        zoom = len(u) / len(ubox)
        reshaped_ubox[:, :, :, i] = scipy.ndimage.zoom(ubox[:, :, :, i], zoom)
    
    u_fluc = u - reshaped_ubox
    k_sgs = 1/2 * (u_fluc[:, :, :, 0]**2 + u_fluc[:, :, :, 1]**2 + u_fluc[:, :, :, 2]**2)
    k_tot = 1/2 * (u[:, :, :, 0]**2  + u[:, :, :, 1]**2 + u[:, :, :, 2]**2)
    
    frac = np.mean(k_sgs / (k_sgs + k_tot))
    
    frac = np.array(frac)
    
    
    print('The fraction between subgrid scale energy and total energy is ' + str(frac))
    
    
    u1u2 = residual_stress[0, 1, :, 0, :].mean()
    dudy = filtered_ugradient[0, 1, :, 0, :].mean()
    
    nu = 5 * 10**(-5)
    
    u_tau = 0.0499   #nu * dudy - u1u2
    y_plus = filtered_ycoor * u_tau / nu
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    
    -----------------------------------------------
    
    PLOT THE PREPROCESSED DATA 
    
    -----------------------------------------------
    
    """
    
    
    if plot == True and dataset_number == 1:
        
        
        x = 2 #round(rn.random()*len(xcoor))
        y = 2 #round(rn.random()*len(ycoor))
        z = 2 #round(rn.random()*len(zcoor))
        
        
        
        
        
        ###############################################
        # R A W     D A T A #
        ###############################################
        
        
        
        #z section
        fig1 = pl.figure(figsize = (20, 20))
        subplot1 = fig1.add_subplot(221)
        a = subplot1.imshow(u[z*filter_size, :, :, 0], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_1$ for $z = ' + str(z) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig1.add_subplot(222)
        b = subplot2.imshow(u[z*filter_size, :, :, 1], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('$u_2$ for $z = ' + str(z) + '$')
        pl.colorbar(b, fraction = 0.045)
        
        subplot3 = fig1.add_subplot(223)
        c = subplot3.imshow(u[z*filter_size, :, :, 2], interpolation = 'none', cmap = cm.jet)
        subplot3.title.set_text('$u_3$ for $z = ' + str(z) + '$')
        pl.colorbar(c, fraction = 0.045)
        
        subplot4 = fig1.add_subplot(224)
        d = subplot4.imshow(p[z*filter_size, :, :, 0], interpolation = 'none', cmap = cm.jet)
        subplot4.title.set_text('$p$ for $z = ' + str(z) + '$')
        pl.colorbar(d, fraction = 0.045)
        
        
        
        
        
        #y section
        fig2 = pl.figure(figsize = (20, 20))
        subplot1 = fig2.add_subplot(221)
        a = subplot1.imshow(u[:, y*filter_size, :, 0], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_1$ for $y = ' + str(y) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig2.add_subplot(222)
        b = subplot2.imshow(u[:, y*filter_size, :, 1], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('$u_2$ for $y = ' + str(y) + '$')
        pl.colorbar(b, fraction = 0.045)
        
        subplot3 = fig2.add_subplot(223)
        c = subplot3.imshow(u[:, y*filter_size, :, 2], interpolation = 'none', cmap = cm.jet)
        subplot3.title.set_text('$u_3$ for $y = ' + str(y) + '$')
        pl.colorbar(c, fraction = 0.045)
        
        subplot4 = fig2.add_subplot(224)
        d = subplot4.imshow(p[:, y*filter_size, :, 0], interpolation = 'none', cmap = cm.jet)
        subplot4.title.set_text('$p$ for $y = ' + str(y) + '$')
        pl.colorbar(d, fraction = 0.045)
        
        
        
        
        #x section
        fig3 = pl.figure(figsize = (20, 20))
        subplot1 = fig3.add_subplot(221)
        a = subplot1.imshow(u[:, :, x*filter_size, 0], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_1$ for $x = ' + str(x) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig3.add_subplot(222)
        b = subplot2.imshow(u[:, :, x*filter_size, 1], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('$u_2$ for $x = ' + str(x) + '$')
        pl.colorbar(b, fraction = 0.045)
        
        subplot3 = fig3.add_subplot(223)
        c = subplot3.imshow(u[:, :, x*filter_size, 2], interpolation = 'none', cmap = cm.jet)
        subplot3.title.set_text('$u_3$ for $x = ' + str(x) + '$')
        pl.colorbar(c, fraction = 0.045)
        
        subplot4 = fig3.add_subplot(224)
        d = subplot4.imshow(p[:, :, x*filter_size, 0], interpolation = 'none', cmap = cm.jet)
        subplot4.title.set_text('$p$ for $x = ' + str(x) + '$')
        pl.colorbar(d, fraction = 0.045)
        
        
        
        
        
        
        ###############################################
        # F I L T E R E D     D A T A #
        ###############################################
        
        
        
        
        #z section
        fig4 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        subplot1 = fig4.add_subplot(331)
        a = subplot1.imshow(u[z*filter_size, :, :, 0], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_1$ for $z = ' + str(z) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot1 = fig4.add_subplot(332)
        a = subplot1.imshow(u[z*filter_size, :, :, 1], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_2$ for $z = ' + str(z) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot1 = fig4.add_subplot(333)
        a = subplot1.imshow(u[z*filter_size, :, :, 2], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_3$ for $z = ' + str(z) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig4.add_subplot(334)
        a = subplot2.imshow(ubox[z, :, :, 0], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Box filter of $u_1$ with filter size = ' + str(filter_size) + ' for $z = ' + str(z) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig4.add_subplot(335)
        a = subplot2.imshow(ubox[z, :, :, 1], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Box filter of $u_2$ with filter size = ' + str(filter_size) + ' for $z = ' + str(z) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig4.add_subplot(336)
        a = subplot2.imshow(ubox[z, :, :, 2], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Box filter of $u_3$ with filter size = ' + str(filter_size) + ' for $z = ' + str(z) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig4.add_subplot(337)
        a = subplot2.imshow(ugauss[z, :, :, 0], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Gaussian filter of $u_1$ with filter size = ' + str(filter_size) + ' for $z = ' + str(z) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig4.add_subplot(338)
        a = subplot2.imshow(ugauss[z, :, :, 1], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Gaussian filter of $u_2$ with filter size = ' + str(filter_size) + ' for $z = ' + str(z) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig4.add_subplot(339)
        a = subplot2.imshow(ugauss[z, :, :, 2], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Gaussian filter of $u_3$ with filter size = ' + str(filter_size) + ' for $z = ' + str(z) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        
        
        
        #y section
        fig5 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        subplot1 = fig5.add_subplot(331)
        a = subplot1.imshow(u[:, y*filter_size, :, 0], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_1$ for $y = ' + str(y) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot1 = fig5.add_subplot(332)
        a = subplot1.imshow(u[:, y*filter_size, :, 1], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_2$ for $y = ' + str(y) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot1 = fig5.add_subplot(333)
        a = subplot1.imshow(u[:, y*filter_size, :, 2], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_3$ for $y = ' + str(y) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig5.add_subplot(334)
        a = subplot2.imshow(ubox[:, y, :, 0], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Box filter of $u_1$ with filter size = ' + str(filter_size) + ' for $y = ' + str(y) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig5.add_subplot(335)
        a = subplot2.imshow(ubox[:, y, :, 1], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Box filter of $u_2$ with filter size = ' + str(filter_size) + ' for $y = ' + str(y) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig5.add_subplot(336)
        a = subplot2.imshow(ubox[:, y, :, 2], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Box filter of $u_3$ with filter size = ' + str(filter_size) + ' for $y = ' + str(y) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig5.add_subplot(337)
        a = subplot2.imshow(ugauss[:, y, :, 0], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Gaussian filter of $u_1$ with filter size = ' + str(filter_size) + ' for $y = ' + str(y) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig5.add_subplot(338)
        a = subplot2.imshow(ugauss[:, y, :, 1], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Gaussian filter of $u_2$ with filter size = ' + str(filter_size) + ' for $y = ' + str(y) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig5.add_subplot(339)
        a = subplot2.imshow(ugauss[:, y, :, 2], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Gaussian filter of $u_3$ with filter size = ' + str(filter_size) + ' for $y = ' + str(y) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        
        
        
        #x section
        fig6 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        subplot1 = fig6.add_subplot(331)
        a = subplot1.imshow(u[:, :, x*filter_size, 0], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_1$ for $x = ' + str(x) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot1 = fig6.add_subplot(332)
        a = subplot1.imshow(u[:, :, x*filter_size, 1], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_2$ for $x = ' + str(x) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot1 = fig6.add_subplot(333)
        a = subplot1.imshow(u[:, :, x*filter_size, 2], interpolation = 'none', cmap = cm.jet)
        subplot1.title.set_text('$u_3$ for $x = ' + str(x) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig6.add_subplot(334)
        a = subplot2.imshow(ubox[:, :, x, 0], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Box filter of $u_1$ with filter size = ' + str(filter_size) + ' for $x = ' + str(x) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig6.add_subplot(335)
        a = subplot2.imshow(ubox[:, :, x, 1], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Box filter of $u_2$ with filter size = ' + str(filter_size) + ' for $x = ' + str(x) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig6.add_subplot(336)
        a = subplot2.imshow(ubox[:, :, x, 2], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Box filter of $u_3$ with filter size = ' + str(filter_size) + ' for $x = ' + str(x) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig6.add_subplot(337)
        a = subplot2.imshow(ugauss[:, :, x, 0], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Gaussian filter of $u_1$ with filter size = ' + str(filter_size) + ' for $x = ' + str(x) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig6.add_subplot(338)
        a = subplot2.imshow(ugauss[:, :, x, 1], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Gaussian filter of $u_2$ with filter size = ' + str(filter_size) + ' for $x = ' + str(x) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        subplot2 = fig6.add_subplot(339)
        a = subplot2.imshow(ugauss[:, :, x, 2], interpolation = 'none', cmap = cm.jet)
        subplot2.title.set_text('Gaussian filter of $u_3$ with filter size = ' + str(filter_size) + ' for $x = ' + str(x) + '$')
        pl.colorbar(a, fraction = 0.045)
        
        
        
        
        
        
        
        ####################################################
        # G R A D I E N T    O F     T H E     F I L T E R #
        ####################################################
        
        
        
        
        #z section
        fig7 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            subplot1 = fig7.add_subplot(3,4,counter)
            a = subplot1.imshow(ubox[z, :, :, i], interpolation = 'none', cmap = cm.jet)
            subplot1.title.set_text('$\overline{u_{' + str(i+1) + '}}$ for $z = ' + str(z) + '$')
            pl.colorbar(a, fraction = 0.045)
            counter += 1
            for j in range(3):
                subplot1 = fig7.add_subplot(3,4,counter)
                a = subplot1.imshow(filtered_ugradient[i, j, z, :, :], interpolation = 'none', cmap = cm.jet)
                subplot1.title.set_text('$\\frac{\partial\overline{u_{' + str(i+1) + '}}}{\partial x_{' + str(j+1) + '}}$ for $z = ' + str(z) + '$')
                pl.colorbar(a, fraction = 0.045)
                counter +=1
        
        
        
        #y section
        fig8 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            subplot1 = fig8.add_subplot(3,4,counter)
            a = subplot1.imshow(ubox[:, y, :, i], interpolation = 'none', cmap = cm.jet)
            subplot1.title.set_text('$\overline{u_{' + str(i+1) + '}}$ for $y = ' + str(y) + '$')
            pl.colorbar(a, fraction = 0.045)
            counter += 1
            for j in range(3):
                subplot1 = fig8.add_subplot(3,4,counter)
                a = subplot1.imshow(filtered_ugradient[i, j, :, y, :], interpolation = 'none', cmap = cm.jet)
                subplot1.title.set_text('$\\frac{\partial\overline{u_{' + str(i+1) + '}}}{\partial x_{' + str(j+1) + '}}$ for $y = ' + str(y) + '$')
                pl.colorbar(a, fraction = 0.045)
                counter +=1
                
                
                
        #x section
        fig9 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            subplot1 = fig9.add_subplot(3,4,counter)
            a = subplot1.imshow(ubox[:, :, x, i], interpolation = 'none', cmap = cm.jet)
            subplot1.title.set_text('$\overline{u_{' + str(i+1) + '}}$ for $x = ' + str(x) + '$')
            pl.colorbar(a, fraction = 0.045)
            counter += 1
            for j in range(3):
                subplot1 = fig9.add_subplot(3,4,counter)
                a = subplot1.imshow(filtered_ugradient[i, j, :, :, x], interpolation = 'none', cmap = cm.jet)
                subplot1.title.set_text('$\\frac{\partial\overline{u_{' + str(i+1) + '}}}{\partial x_{' + str(j+1) + '}}$ for $x = ' + str(x) + '$')
                pl.colorbar(a, fraction = 0.045)
                counter +=1
                
        
        
        
        
        
        
        
        ####################################################
        # S T R A I N    T E N S O R #
        ####################################################
        
        
        
        
        
        
        #z section
        fig10 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            for j in range(3):
                subplot1 = fig10.add_subplot(3,3,counter)
                a = subplot1.imshow(strain_tensor[i, j, z, :, :], interpolation = 'none', cmap = cm.jet)
                subplot1.title.set_text('$\overline{S_{' + str(i+1) + str(j+1) + '}}$ for $z = ' + str(z) + '$')
                pl.colorbar(a, fraction = 0.045)
                counter += 1
        
        
        #y section
        fig11 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            for j in range(3):
                subplot1 = fig11.add_subplot(3,3,counter)
                a = subplot1.imshow(strain_tensor[i, j, :, y, :], interpolation = 'none', cmap = cm.jet)
                subplot1.title.set_text('$\overline{S_{' + str(i+1) + str(j+1) + '}}$ for $y = ' + str(y) + '$')
                pl.colorbar(a, fraction = 0.045)
                counter += 1
        
        
        #x section
        fig12 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            for j in range(3):
                subplot1 = fig12.add_subplot(3,3,counter)
                a = subplot1.imshow(strain_tensor[i, j, :, :, x], interpolation = 'none', cmap = cm.jet)
                subplot1.title.set_text('$\overline{S_{' + str(i+1) + str(j+1) + '}}$ for $x = ' + str(x) + '$')
                pl.colorbar(a, fraction = 0.045)
                counter += 1
        
        
        
        
        
        
        ####################################################
        # R E S I D U A L    S T R E S S    T E N S O R #
        ####################################################
        
        
        
        
        
        
        #z section
        fig13 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            for j in range(3):
                subplot1 = fig13.add_subplot(3, 3, counter)
                a = subplot1.imshow(residual_stress[i, j, z, :, :], interpolation = 'none', cmap = cm.jet)
                subplot1.title.set_text('$\overline{u\'_' + str(i+1) + 'u\'_' + str(j+1) + '}$ for z = ' + str(z))
                pl.colorbar(a, fraction = 0.045)
                
                counter += 1
        
        
        #y section
        fig14 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            for j in range(3):
                subplot1 = fig14.add_subplot(3, 3, counter)
                a = subplot1.imshow(residual_stress[i, j, :, y, :], interpolation = 'none', cmap = cm.jet)
                subplot1.title.set_text('$\overline{u\'_' + str(i+1) + 'u\'_' + str(j+1) + '}$ for y = ' + str(y))
                pl.colorbar(a, fraction = 0.045)
                
                counter += 1
        
        
        #x section
        fig15 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        counter = 1
        for i in range(3):
            for j in range(3):
                subplot1 = fig15.add_subplot(3, 3, counter)
                a = subplot1.imshow(residual_stress[i, j, :, :, x], interpolation = 'none', cmap = cm.jet)
                subplot1.title.set_text('$\overline{u\'_' + str(i+1) + 'u\'_' + str(j+1) + '}$ for x = ' + str(x))
                pl.colorbar(a, fraction = 0.045)
                
                counter += 1
        
        
        
        
        
        ########################################################################################
        # G R A D I E N T     O F     T H E      R E S I D U A L    S T R E S S    T E N S O R #
        ########################################################################################
        
        
        
        
        
        
        
        
        #plot all components of the stress tensor
        fig16 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.05, bottom=0.01, right=0.95, top=0.99, wspace=0.85, hspace=0.2)
        counter = 1
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    subplot1 = fig16.add_subplot(3, 9, counter)
                    a = subplot1.imshow(residual_stress_gradient[i, j, k, z, :, :], interpolation = 'none', cmap = cm.jet)
                    subplot1.title.set_text('$\\frac{\partial\overline{u\'_' + str(i+1) + 'u\'_' + str(j+1) + '}}{\partial x_' + str(k+1) + '}$ for z = ' + str(z))
                    pl.colorbar(a, fraction = 0.045)
                    
                    counter += 1
        
        
        #plot all components of the stress tensor
        fig17 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.05, bottom=0.01, right=0.95, top=0.99, wspace=0.85, hspace=0.2)
        counter = 1
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    subplot1 = fig17.add_subplot(3, 9, counter)
                    a = subplot1.imshow(residual_stress_gradient[i, j, k, :, y, :], interpolation = 'none', cmap = cm.jet)
                    subplot1.title.set_text('$\\frac{\partial\overline{u\'_' + str(i+1) + 'u\'_' + str(j+1) + '}}{\partial x_' + str(k+1) + '}$ for y = ' + str(y))
                    pl.colorbar(a, fraction = 0.045)
                    
                    counter += 1
        
        
        #plot all components of the stress tensor
        fig18 = pl.figure(figsize = (20, 20))
        pl.subplots_adjust(left=0.05, bottom=0.01, right=0.95, top=0.99, wspace=0.85, hspace=0.2)
        counter = 1
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    subplot1 = fig18.add_subplot(3, 9, counter)
                    a = subplot1.imshow(residual_stress_gradient[i, j, k, :, :, x], interpolation = 'none', cmap = cm.jet)
                    subplot1.title.set_text('$\\frac{\partial\overline{u\'_' + str(i+1) + 'u\'_' + str(j+1) + '}}{\partial x_' + str(k+1) + '}$ for x = ' + str(x))
                    pl.colorbar(a, fraction = 0.045)
                    
                    counter += 1
        
        
        
        
        
        #save all figures
        
        directory = 'channel5200_' + str(dataset_number) + '_figures/'
        
        dir1 = directory + str('u_and_p_for_z=') + str(z)
        fig1.savefig(dir1)
        dir2 = directory + str('u_and_p_for_y=') + str(y)
        fig2.savefig(dir2)
        dir3 = directory + str('u_and_p_for_x=') + str(x)
        fig3.savefig(dir3)
        dir4 = directory + str('filtered_u_for_z=') + str(z)
        fig4.savefig(dir4)
        dir5 = directory + str('filtered_u_for_y=') + str(y)
        fig5.savefig(dir5)
        dir6 = directory + str('filtered_u_for_x=') + str(x)
        fig6.savefig(dir6)
        dir7 = directory + str ('gradient_of_filtered_u_for_z=') + str(z)
        fig7.savefig(dir7)
        dir8 = directory + str ('gradient_of_filtered_u_for_y=') + str(y)
        fig8.savefig(dir8)
        dir9 = directory + str ('gradient_of_filtered_u_for_x=') + str(x)
        fig9.savefig(dir9)
        dir10 = directory + str ('strain_tensor_for_z=') + str(z)
        fig10.savefig(dir10)
        dir11 = directory + str ('strain_tensor_for_y=') + str(y)
        fig11.savefig(dir11)
        dir12 = directory + str ('strain_tensor_for_x=') + str(x)
        fig12.savefig(dir12)
        dir13 = directory + str ('residual_stress_tensor_for_z=') + str(z)
        fig13.savefig(dir13)
        dir14 = directory + str ('residual_stress_tensor_for_y=') + str(y)
        fig14.savefig(dir14)
        dir15 = directory + str ('residual_stress_tensor_for_x=') + str(x)
        fig15.savefig(dir15)
        dir16 = directory + str ('gradient_of_the_residual_stress_tensor_for_z=') + str(z)
        fig16.savefig(dir16)
        dir17 = directory + str ('gradient_of_the_residual_stress_tensor_for_y=') + str(y)
        fig17.savefig(dir17)
        dir18 = directory + str ('gradient_of_the_residual_stress_tensor_for_x=') + str(x)
        fig18.savefig(dir18)
    
    
    
    
    
    
    
    
    """
    
    -----------------------------------------------
    
    SAVING THE DATA IN AN HDF5 FILE 
    
    -----------------------------------------------
    
    """
    
    U = np.ndarray((3, len(u), len(u), len(u)))
    Ubox = np.ndarray((3, len(ubox), len(ubox), len(ubox)))
    Ugauss = np.ndarray((3, len(ugauss), len(ugauss), len(ugauss)))
    
    for i in range(3):
        U[i, :, :, :] = u[:, :, :, i]
        Ubox[i, :, :, :] = ubox[:, :, :, i]
        Ugauss[i, :, :, :] = ugauss[:, :, :, i]
    
    
    
    
    filename = 'channel5200_' + str(dataset_number) + '_processed.h5'
    preprocessed_data = h5py.File(filename, 'w')
    preprocessed_data.create_dataset('Velocity_0001', data = U)
    preprocessed_data.create_dataset('Pressure_0001', data = p)
    preprocessed_data.create_dataset('xcoor', data = xcoor)
    preprocessed_data.create_dataset('ycoor', data = ycoor)
    preprocessed_data.create_dataset('zcoor', data = zcoor)
    preprocessed_data.create_dataset('filtered_xcoor', data = filtered_xcoor)
    preprocessed_data.create_dataset('filtered_ycoor', data = filtered_ycoor)
    preprocessed_data.create_dataset('filtered_zcoor', data = filtered_zcoor)
    preprocessed_data.create_dataset('nondimensional_ycoor', data = y_plus)
    preprocessed_data.create_dataset('Velocity_Gaussian_filter', data = Ugauss)
    preprocessed_data.create_dataset('Velocity_Box_filter', data = Ubox)
    preprocessed_data.create_dataset('Gradient_of_filtered_velocity', data = filtered_ugradient)
    preprocessed_data.create_dataset('Strain_tensor', data = strain_tensor)
    preprocessed_data.create_dataset('Residual_stress', data = residual_stress)
    preprocessed_data.create_dataset('Residual_stress_gradient', data = residual_stress_gradient)
    preprocessed_data.create_dataset('Fraction_of_subgrid_scale_energy', data = frac)
    preprocessed_data.create_dataset('Filter_size', data = filter_size)
    preprocessed_data.close()