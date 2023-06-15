/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) YEAR OpenFOAM Foundation
     \\/     M anipulation  |
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

\*---------------------------------------------------------------------------*/

#include "codedFixedValueFvPatchFieldTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
//{{{ begin codeInclude

//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    // dynamicCode:
    // SHA1 = 1353377008e737c2acc9c5068603a09564d5dd41
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void sinFixedValue_1353377008e737c2acc9c5068603a09564d5dd41(bool load)
    {
        if (load)
        {
            // code that can be explicitly executed after loading
        }
        else
        {
            // code that can be explicitly executed before unloading
        }
    }
}

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

makeRemovablePatchTypeField
(
    fvPatchVectorField,
    sinFixedValueFixedValueFvPatchVectorField
);


const char* const sinFixedValueFixedValueFvPatchVectorField::SHA1sum =
    "1353377008e737c2acc9c5068603a09564d5dd41";


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

sinFixedValueFixedValueFvPatchVectorField::
sinFixedValueFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF)
{
    if (false)
    {
        Info<<"construct sinFixedValue sha1: 1353377008e737c2acc9c5068603a09564d5dd41"
            " from patch/DimensionedField\n";
    }
}


sinFixedValueFixedValueFvPatchVectorField::
sinFixedValueFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict)
{
    if (false)
    {
        Info<<"construct sinFixedValue sha1: 1353377008e737c2acc9c5068603a09564d5dd41"
            " from patch/dictionary\n";
    }
}


sinFixedValueFixedValueFvPatchVectorField::
sinFixedValueFixedValueFvPatchVectorField
(
    const sinFixedValueFixedValueFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper)
{
    if (false)
    {
        Info<<"construct sinFixedValue sha1: 1353377008e737c2acc9c5068603a09564d5dd41"
            " from patch/DimensionedField/mapper\n";
    }
}


sinFixedValueFixedValueFvPatchVectorField::
sinFixedValueFixedValueFvPatchVectorField
(
    const sinFixedValueFixedValueFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF)
{
    if (false)
    {
        Info<<"construct sinFixedValue sha1: 1353377008e737c2acc9c5068603a09564d5dd41 "
            "as copy/DimensionedField\n";
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

sinFixedValueFixedValueFvPatchVectorField::
~sinFixedValueFixedValueFvPatchVectorField()
{
    if (false)
    {
        Info<<"destroy sinFixedValue sha1: 1353377008e737c2acc9c5068603a09564d5dd41\n";
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void sinFixedValueFixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        Info<<"updateCoeffs sinFixedValue sha1: 1353377008e737c2acc9c5068603a09564d5dd41\n";
    }

//{{{ begin code
    #line 66 "/home/kacperjanczuk/Desktop/MLTurbulence/a_posteriori/transition_box/0/U/boundaryField/inlet"
const scalar freq = 0.0127;
        const scalar A = 0.05;
        const fvPatch& boundaryPatch = patch(); 
        const vectorField& Cf = boundaryPatch.Cf();
        const scalar t = this->db().time().value();

        vectorField& field= *this;  
        forAll(Cf,faceI) 
        {
          scalar z = Cf[faceI].z(); // coordinate z face
          field[faceI] = vector(0, A*sin(6.28/240*z)*sin(6.28*freq*t) +A/2*sin(6.28/120*z+40)*sin(6.28*freq*t) +A/4*sin(6.28/60*z+10)*sin(6.28*freq*t) +A/8*sin(6.28/30*z+6)*sin(6.28*freq*t) +A/16*sin(6.28/15*z+32)*sin(6.28*freq*t) +A/32*sin(6.28/8*z+12)*sin(6.28*freq*t) +A/64*sin(6.28/4*z+4)*sin(6.28*freq*t) ,0);
      //    field[faceI] = vector(0, 20 ,0);
        }
//}}} end code

    this->fixedValueFvPatchField<vector>::updateCoeffs();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

