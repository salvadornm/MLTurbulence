// Solve the Momentum equation

MRF.correctBoundaryVelocity(U);

// Info << 17 << endl;

fvVectorMatrix UEqn
(
    fvm::ddt(U) + fvm::div(phi, U)
  + MRF.DDt(U)
  + turbulence->divDevReff(U)
//   - fvm::laplacian(this->rho_*this->nu, U)
//   - fvm::laplacian(this->nu, U)
  + fvc::div(Tau) // This is the divergence of ANN-generated residual stress term
 ==
    fvOptions(U)
);

// Info << 18 << endl;

UEqn.relax();

// Info << 19 << endl;

fvOptions.constrain(UEqn);

// Info << 20 << endl;

if (piso.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    fvOptions.correct(U);
}

// Info << 21 << endl;