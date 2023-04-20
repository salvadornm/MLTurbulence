# Setting up the simulation

This section describes how the simulation is set up and where specific settings are specified.



## Set up a regular simulation

1. Copy the existing LES channel flow simulation from `tutorials/incompressible/pimpleFoam/LES/channel` and rename it `channel_pisoFoamNN`.

2. In `system/controlDict` specify the type of solver, the start and end time of the simulation, the time step and the writing interval.

```C++
application     pisoFoam;

startTime       0;

endTime         0.4;

writeInterval   1;
```


3. In `system/blockMeshDict` create the domain and set up the seed to mesh it. Specify the vertices:

```C++
vertices
(
    (0 0 0)
    (4 0 0)
    (0 1 0)
    (4 1 0)
    (0 2 0)
    (4 2 0)
    (0 0 2)
    (4 0 2)
    (0 1 2)
    (4 1 2)
    (0 2 2)
    (4 2 2)
);
```
Create the corresponding blocks (2 for channel flow as there will be refinement closer to the wall).

```C++
blocks
(
    \\ type of block    (vertices forming in the block)     (number of cells in each direction)     type of refinement      (growth rate in each direction)
    hex (0 1 3 2 6 7 9 8) (40 25 30) simpleGrading (1 10.7028 1)
    hex (2 3 5 4 8 9 11 10) (40 25 30) simpleGrading (1 0.0934 1)
);
```

Specify the boundary conditions for each block face.

```C++
boundary
(
    bottomWall
    {
        type            wall;
        faces           ((0 1 7 6));
    }
);
```

4. In `system/decomposeParDict` set the type of parallelisation to `simple` and specify the amount of blocks in each direction.

```C++
method          simple;

simpleCoeffs
{
    n           (1 2 2);
    delta       0.001;
}
```


5. In `system/fvSolution` change the corrector algorithm to PISO.

```C++
PISO
{
    nOuterCorrectors 1;
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        1001;
    pRefValue       0;
}

```



6. In `constant/momentumTransport` set the closure model to Smagorinsky and specify the constants.


```C++
model           Smagorinsky;


SmagorinskyCoeffs
{
    Ck      0.094;
    Ce      1.048;
}
```




7. Create the mesh by using the command line in the simulation directory and running

```
blockMesh
```

8. Decompose the domain to run in parallel by using the command line in the simulation directory and running

```
decomposePar
```

9. Run the simulation in parrallel using

```
mpirun -np 4 pisoFoam -parallel
```

10. Reconstruct the domain by using

```
reconstructPar
```

11. Visualise the results in `ParaView` using the command

```
paraFoam
```

12. to delete the simulation results use the command

```
foamListTimes -rm
```








## Use perturbU


1. To use `perturbU` one must use OpenFOAM vs 8. Hence change using



2. In `constant/transportProperties` double the label for Ubar and nu due to a bug

```C++
Ubar Ubar           [0 1 -1 0 0 0 0] (0.1335 0 0);

nu nu             [0 2 -1 0 0 0 0] 2e-05;
```


3. In `constant` copy the `perturbUDict` file and specify the desired turbulent properties.


4. In `0/U` create a uniform field with the velocity equal to the inlet velocity


```C++
internalField   uniform (0.1335, 0, 0);
```

5. After meshing run the following command to perturb the velocity field

```
perturbU
```







## Use pisoFoamNN

1. In `system/controlDict` change the solver name

```C++
application     pisoFoamNN;
```



2. In `system/fvSchemes` specify the numerical scheme for the divergence of TauNN.


```C++
divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;
    div(phi,k)      Gauss limitedLinear 1;
    div(phi,B)      Gauss limitedLinear 1;
    div(B)          Gauss linear;
    div(phi,nuTilda) Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
    div(TauNN)        Gauss linear;
}
```


3. In `constant/momentumTransport` set the Smagorinsky constant to a low value by setting


```C++
model           Smagorinsky;


SmagorinskyCoeffs
{
    Ck      0.00001;
    Ce      1.048;
}
```



4. In the `0` directory create a new field function `TauNN` by specifying its name and class.

```C++
FoamFile
{
    version     2.0;
    format      ascii;
    class       volTensorField;
    location    "1";
    object      TauNN;
}
```

Specify the units.

```C++
dimensions      [0 2 -2 0 0 0 0];
```


Specify the initial field.

```C++
internalField   uniform (
		 0.001 0 0
		 0 0.001 0
		 0 0 0.001 
		 );
```


Specify all boundaries.


```C++
boundaryField
{
    bottomWall
    {
        type    fixedValue;
        value   uniform   (0 0 0 0 0 0 0 0 0);

    }
    topWall
    {
        type    fixedValue;
        value   uniform   (0 0 0 0 0 0 0 0 0);
    }
    sides1_half0
    {
        type            cyclic;
    }
    sides2_half0
    {
        type            cyclic;
    }
    inout1_half0
    {
        type            cyclic;
    }
    inout2_half0
    {
        type            cyclic;
    }
    sides2_half1
    {
        type            cyclic;
    }
    sides1_half1
    {
        type            cyclic;
    }
    inout1_half1
    {
        type            cyclic;
    }
    inout2_half1
    {
        type            cyclic;
    }
}
```



5. Run the simulation.
