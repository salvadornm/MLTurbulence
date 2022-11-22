/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    pisoFoam

Group
    grpIncompressibleSolvers

Description
    Transient solver for incompressible, turbulent flow, using the PISO
    algorithm.

    \heading Solver details
    The solver uses the PISO algorithm to solve the continuity equation:

        \f[
            \div \vec{U} = 0
        \f]

    and momentum equation:

        \f[
            \ddt{\vec{U}} + \div \left( \vec{U} \vec{U} \right) - \div \gvec{R}
          = - \grad p
        \f]

    Where:
    \vartable
        \vec{U} | Velocity
        p       | Pressure
        \vec{R} | Stress tensor
    \endvartable

    Sub-models include:
    - turbulence modelling, i.e. laminar, RAS or LES
    - run-time selectable MRF and finite volume options, e.g. explicit porosity

    \heading Required fields
    \plaintable
        U       | Velocity [m/s]
        p       | Kinematic pressure, p/rho [m2/s2]
        \<turbulence fields\> | As required by user selection
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "tf_utils.H"
// #include tensorflow/c/c_api.h>
// #include <torch/torch.h>
// #include "~/libtorch/include/torch/csrc/api/include/torch/torch.h"
// #include "torch/torch.h"

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pisoControl.H"
#include "fvOptions.H"
#include <fstream>
#include <iostream>
#include <cstdlib>

// #include "keras2cpp/src/model.h"



// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{





    argList::addNote
    (
        "Transient solver for incompressible, turbulent flow,"
        " using the PISO algorithm."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


    // g.open("graph.pb", std::ios::binary | std::ios::in);
    // g.open("latest.pb", std::ios::binary);

    Info<< "\nStarting time loop\n" << endl;

    Info << "Opening neural network" << endl;

    int timeStep_n = 0;

    float totalTauEqnTime = 0;

    float totalANNTime = 0;


    // std::ifstream g;
    // g.open("/usr/include/test.txt", std::fstream::in);

    // if (g.fail() || !g.is_open()) {
    // Info << "g.fail() or !g.is_open()" << endl;
    // } else {
    //     Info << "Appears to have worked" << endl;
    // }

    // string getcontent;
    // while(! g.eof())
    // {
    //     g >> getcontent;
    //     Info << getcontent << endl;
    // }


    
    // ifstream myfile ("test.txt");
    // Info << myfile << endl;
    // #include "neuralNetwork.H"

    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        #include "CourantNo.H"

        // Pressure-velocity PISO corrector
        {
            #include "TauEqn.H"
            Info << "Found Tau" << endl;
            Info << "UEqn..." << endl;
            #include "UEqn.H"
            Info<< "Done UEqn, doing piso:" << nl << endl;

            // --- PISO loop
            while (piso.correct())
            {
                #include "pEqn.H"
            }
            Info<< "Done piso" << nl << endl;
        }

        laminarTransport.correct();
        turbulence->correct();

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
