# Set up the closure model

## Install tensorflow utilities

Install the tensorflow utilities for `C` and `C++` for Ubuntu from: https://www.tensorflow.org/install/lang_c
 



 ## Set up the PISO algorithm

 1. Copy the standard PISO algorithm directory from the `.\OpenFOAM\src\pisoFoam` directory into any preferred location and change the name to `pisoFoamNN`.

2. Modify the `Make\` directory by changing the `Files` file by adding the following lines

```
pisoFoamNN.C
tf_utils.C

EXE = $(FOAM_USER_APPBIN)/pisoFoamNN
```

3. Modify the `Make\` directory by changing the `Options` file by adding the following lines

```
EXE_INC = \
    -I$(LIB_SRC)/MomentumTransportModels/momentumTransportModels/lnInclude \
    -I$(LIB_SRC)/MomentumTransportModels/incompressible/lnInclude \
    -I$(LIB_SRC)/physicalProperties/lnInclude \
    -I$(LIB_SRC)/finiteVolume/lnInclude \
    -I$(LIB_SRC)/meshTools/lnInclude \
    -I$(LIB_SRC)/sampling/lnInclude

EXE_LIBS = \
    -lmomentumTransportModels \
    -lincompressibleMomentumTransportModels \
    -lphysicalProperties \
    -lfiniteVolume \
    -lmeshTools \
    -lfvModels \
    -lfvConstraints \
    -lsampling \
    -ltensorflow
```

4. Create the residual stress tensor field in the `createFields.H` file by adding the following code:

```C++
volTensorField TauNN
(
    IOobject
    (
        "TauNN",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);
```


5. Create the `TauEqn.H` file to compute the residual stresses using the NN. Here the values of the single input functions are collected from the previous fields and are scaled. The network architecture is opened. Then the inputs are fed through the network to obtain the residual stresses. Finally these are rescaled and saved as the residual stress field.


6. Modify the `UEqn.H` file accordingly to include the divergence of the residual stress tensor field and to use the correct function to compute the divergence of U.

```C++
fvVectorMatrix UEqn
(
    fvm::ddt(U) + fvm::div(phi, U)
  + MRF.DDt(U)
  + turbulence->divDevSigma(U)          \\ alternatively use divDevReff(U)
  + fvc::div(TauNN)
 ==
    fvModels.source(U)
);

```


7. Rename `pisoFoam.C` to `pisoFoamNN.C`.

8. Modify `pisoFoamNN.C` by adding the necessary libraries:

```C++
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "tf_utils.H"       // include tensorflow libraries
```


9. Modify `pisoFoam.C` by including `TauEqn.H` in the PISO corrector portion of the code as follows:

```C++
// Pressure-velocity PISO corrector
{
    fvModels.correct();

    //[NEW] added Tensor equation with NN
    #include "TauEqn.H"   

    #include "UEqn.H"

    // --- PISO loop
    while (piso.correct())
    {
        #include "pEqn.H"
    }
}
```



10. run the command `wclean` through the command line when in the directory to clean up the directory of the solver.

11. run the command `wmake` through the command line when in the directory to compile the solver.