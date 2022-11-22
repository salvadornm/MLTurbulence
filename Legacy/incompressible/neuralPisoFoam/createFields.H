Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

volTensorField Tau
(
		IOobject
		(
			"Tau",
			runTime.timeName(),
			mesh,
			IOobject::NO_READ,
			IOobject::AUTO_WRITE
		),
		mesh,
		// dimensionedTensor
		// (
		// 	"zero",
		// 	(0,2,-2,0,0,0,0)
		// 	// tensor::zero
		// )
        dimArea/dimTime/dimTime
	);

volTensorField gradientUWrite
(
		IOobject
		(
			"gradientUWrite",
			runTime.timeName(),
			mesh,
			IOobject::NO_READ,
			IOobject::AUTO_WRITE
		),
		mesh,
        dimless/dimTime
);

// volTensorField backscatter
// (
// 		IOobject
// 		(
// 			"backscatter",
// 			runTime.timeName(),
// 			mesh,
// 			IOobject::NO_READ,
// 			IOobject::AUTO_WRITE
// 		),
// 		mesh,
//         dimless/dimTime
// );

volScalarField subGridDissipation
(
    IOobject
    (
        "subGridDissipation",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

// volScalarField neg2strainRateTensorSq
// (
//     IOobject
//     (
//         "subGridDissipation",
//         runTime.timeName(),
//         mesh,
//         IOobject::NO_READ,
//         IOobject::NO_WRITE
//     ),
//     mesh
// );

#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, piso.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"
