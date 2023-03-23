# A Priori codes

The codes in this folder deal with the data pre-processing and with the development and training of the NN.


## `data_preprocesing_code.py`

The code takes datacubes in the hdf5 format containing velocity and pressure data, filters it, according to a prespecified filter width, and computes a series of derived variables. These are saved again in a cubic data array in hdf5 format. The output file contains:

- Velocity
- Pressure
- Original spatial coordinates
- Filtered spatial coordinates
- Filtered velocity
- Filtered velocity gradient
- Filtered strain tensor
- Non-dimensional wall distance
- Residual stress tensor
- Residual stress tensor divergence
- Fraction of sub-grid scale kinetic energy
- Filter size

All computed variables can be plotted by setting

```python
  plot = True
```

## `NN code.py`


### Main Functions

This code reads the pre-processed datacubes in hdf5 format, trains the network and validates it a priori.

Training and testing files are specified in the arrays

```python
  train_files = ['example_train_file.h5']
  test_files = ['example_test_file.h5']
```

The `shape_train_data` function receives the datacubes, flattens and normalises the data, and returns the input and labelled output arrays, as well as the scaling parameters.

The `shape_test_data` function receives the datacubes as well as the scaling parameters, flattens and normalises, and returns the input and labelled output.

The `correlation_coefficient` function receives the predicted output and the true label and computes the Pearson correlation factor.

The training data is further split into actual training and validation data using the `train_test_split` function and by specifying the fraction in 

```python
test_split = 0.2
```


### Network architecture and training

The training session is initialised by clearing any previous session using

```python
keras.backend.clear_session()
```

The model is then created using the standard `keras.Sequential` function and by specifying the number of neurons per layer and the type of layer.

```python
#create the ML model by stacking layers
model = tf.keras.Sequential([
    keras.layers.Dense(240, activation = activation_function, input_dim = 20),
    keras.layers.Dense(120, activation = activation_function),
    keras.layers.Dense(240, activation = activation_function),
    keras.layers.Dense(6)
])
```

A model summary is then printed using 

```python
model.summary()
```

The model is then compiled using

```python
model.compile( 
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=loss_function, 
              metrics = [cc_0, cc_1, cc_2, cc_3, cc_4, cc_5])
```

The correlation coefficients for each residual stress term are used as metric for the network during training.

Finally the network is trained by fitting the training data.

```python
history = model.fit(train_input,
                    train_output,
                    validation_data = (val_input, val_output),
                    epochs=n_epochs,
                    verbose = 1)
```

The main parameters to change the network are specified at the top of the code:

```python
n_epochs = 100
activation_function = 'tanh'
layer_MLPs = [480, 240, 480, 6]
loss_function = 'mse'
learning_rate = 0.001
```

### Saving

- The network is then saved in the `hdf5` format.

- The model history is saved in the `json` format.

- The scaling parameters are saved in the `hdf5` format.

- A general description of the network, the training and testing files and the network name are saved as a `txt` file.




### Network testing

The testing files are opened and processed via the `shape_test_data`, and a prediction is made using the `model.predict` function using the previously trained model. The relative error between the true and predicted outputs are calculated.

The flattened arrays are reconstructed into 3D arrays. Smagorinsky stresses are computed as a measure of comparison using;

```python
for m in range(3):
    for n in range(3):
        if m == n:
            smag_stress[m, n, k, j, i] = -3 * nu_t * strain_tensor[m, n, k, j, i]
        else:
            smag_stress[m, n, k, j, i] = -2 * nu_t * strain_tensor[m, n, k, j, i]
```


### Plotting and saving plots

A series of data sections are plotted for the true, predicted and smagorinsky stresses, as well as the correlation coefficient and MSE as the network is trained.

Saving is enabled via

```python
save = True
```



## `NN_export.py`


This code takes the trained model and saves it in the `pb` format.




## `model_post_analysis.py`

This code opens the trained network and a series of specified testing data arrays and uses them to validate the model itself. All plots are the same as in `NN_code.py`.


## `y_profile plotting.py`

This code opens the pre processed data cubes, calculates and plots the non-dimensional wall velocity profile.