// Run the neural network to find the Tau field

timeStep_n = timeStep_n + 1;

auto startTauEqnTime = std::chrono::system_clock::now();

volTensorField gradientUfull = fvc::grad(U);

const tensorField& gradientU = gradientUfull.internalField();
// gradientU = gradientUfull.internalField();
int numCells = mesh.cells().size(); //This works, returns expected value
for (int cellI = 0; cellI < numCells; cellI++)
{
    gradientUWrite[cellI][0] = gradientU[cellI][0];
    gradientUWrite[cellI][1] = gradientU[cellI][1];
    gradientUWrite[cellI][2] = gradientU[cellI][2];
    gradientUWrite[cellI][3] = gradientU[cellI][3];
    gradientUWrite[cellI][4] = gradientU[cellI][4];
    gradientUWrite[cellI][5] = gradientU[cellI][5];
    gradientUWrite[cellI][6] = gradientU[cellI][6];
    gradientUWrite[cellI][7] = gradientU[cellI][7];
    gradientUWrite[cellI][8] = gradientU[cellI][8];

}

// strainRateTensor = symm(fvc::grad(U));


// Info<< "1" << nl << endl;

const int numInputs = 9;
const int numOutputs = 6;

const std::vector<std::int64_t> inputDims = {numCells,numInputs};

// Info<< "2" << nl << endl;

// volTensorField Tau = gradientUfull;

float gradOffset = 0;
float gradScale = 1;
int gradientUCount = 0;
std::vector<float> inputVals_;
forAll(gradientU,cellI)
{
    const tensor L = gradientU[cellI];

    float g0 = (L.xx() - gradOffset) / gradScale;
    float g1 = (L.xy() - gradOffset) / gradScale;
    float g2 = (L.xz() - gradOffset) / gradScale;
    float g3 = (L.yx() - gradOffset) / gradScale;
    float g4 = (L.yy() - gradOffset) / gradScale;
    float g5 = (L.yz() - gradOffset) / gradScale;
    float g6 = (L.zx() - gradOffset) / gradScale;
    float g7 = (L.zy() - gradOffset) / gradScale;
    float g8 = (L.zz() - gradOffset) / gradScale;

    inputVals_.push_back(g0);
    inputVals_.push_back(g1);
    inputVals_.push_back(g2);
    inputVals_.push_back(g3);
    inputVals_.push_back(g4);
    inputVals_.push_back(g5);
    inputVals_.push_back(g6);
    inputVals_.push_back(g7);
    inputVals_.push_back(g8);

    if (cellI == 0) {
        Info << "Grad: " << inputVals_ << endl;
    }

    gradientUCount += 1;

}

// Info<< "3" << nl << endl;

auto input_tensor_ = tf_utils::CreateTensor(TF_FLOAT, inputDims, inputVals_);
TF_Tensor* output_tensor_ = nullptr;

// Info<< "4" << nl << endl;

auto graph_ = tf_utils::LoadGraph("/home/george/OpenFOAM/george-v2006/applications/solvers/incompressible/neuralPisoFoam/24_05_21_NN01.pb"); // Main one is 29_05_21
if (graph_ == nullptr) {
    Info << "Can't load graph" << endl;
}

// Info<< "5" << nl << endl;

auto input_op = TF_Output{TF_GraphOperationByName(graph_,"dense_input"), 0};
auto out_op = TF_Output{TF_GraphOperationByName(graph_,"dense_5/BiasAdd"), 0};

// Info<< "6" << nl << endl;

auto status_ = TF_NewStatus();
auto options_ = TF_NewSessionOptions();
auto sess_ = TF_NewSession(graph_, options_, status_);

// Info<< "7" << nl << endl;

auto startANNTime = std::chrono::system_clock::now();

TF_SessionRun(sess_,
            nullptr, // Run options.
            &input_op, &input_tensor_, 1, // Input tensor ops, input tensor values, number of inputs.
            &out_op, &output_tensor_, 1, // Output tensor ops, output tensor values, number of outputs.
            nullptr, 0, // Target operations, number of targets.
            nullptr, // Run metadata.
            status_ // Output status.
            );

auto endANNTime = std::chrono::system_clock::now();
std::chrono::duration<double> ANNTime = endANNTime - startANNTime;

// totalTauEqnTime = totalTauEqnTime + totalANNTime.count();
totalANNTime = totalANNTime + ANNTime.count();
double averageANNTime = totalANNTime / timeStep_n;

Info << "Average ANN time taken: " << averageANNTime << endl;
// std::cout << "elapsed time: " << ANNTime.count() << "s\n";

auto data = static_cast<float*>(TF_TensorData(output_tensor_));

// Info<< "8" << nl << endl;
// const volScalarField nu_t = turbulence->nut();
// for (int cellI = 0; cellI < numCells; cellI++)
// {

//     subGridDissipation[cellI] = 0;

//     // Info << strainRateTensor[cellI] << endl;
//     for (int index = 0; index < 9; index++){

//         int symmIndex = index;
//         switch(index){
//             case 3:
//                 symmIndex = 1;
//                 break;
//             case 4:
//                 symmIndex = 3;
//                 break;
//             case 5:
//                 symmIndex = 4;
//                 break;
//             case 6:
//                 symmIndex = 2;
//                 break;
//             case 7:
//                 symmIndex = 4;
//                 break;
//             case 8:
//                 symmIndex = 5;
//                 break;
//             default:
//                 symmIndex = index;
//         }

//         subGridDissipation[cellI] = subGridDissipation[cellI] - nu_t[cellI] * 2 * sqr(symm(gradientU[cellI])[symmIndex]);
//     }
// }

// float tauOffset = 0;
// float tauScale = 1;
for (int i = 0; i < numCells; i++)
{

    Tau[i][0] = data[numOutputs*i+0]; //*tauScale + tauOffset;
    Tau[i][1] = data[numOutputs*i+3]; //*tauScale + tauOffset;
    Tau[i][2] = data[numOutputs*i+4]; //*tauScale + tauOffset;
    Tau[i][3] = data[numOutputs*i+3]; //*tauScale + tauOffset;
    Tau[i][4] = data[numOutputs*i+1]; //*tauScale + tauOffset;
    Tau[i][5] = data[numOutputs*i+5]; //*tauScale + tauOffset;
    Tau[i][6] = data[numOutputs*i+4]; //*tauScale + tauOffset;
    Tau[i][7] = data[numOutputs*i+5]; //*tauScale + tauOffset;
    Tau[i][8] = data[numOutputs*i+2]; //*tauScale + tauOffset;

    subGridDissipation[i] = 0;

    for (int index = 0; index < 9; index++){

        int symmIndex = index;
        switch(index){
            case 3:
                symmIndex = 1;
                break;
            case 4:
                symmIndex = 3;
                break;
            case 5:
                symmIndex = 4;
                break;
            case 6:
                symmIndex = 2;
                break;
            case 7:
                symmIndex = 4;
                break;
            case 8:
                symmIndex = 5;
                break;
            default:
                symmIndex = index;
        }

        subGridDissipation[i] = subGridDissipation[i] + symm(gradientU[i])[symmIndex] * Tau[i][index];
    }

    if (i == 0) {
        Info << "Tau: " << data[numOutputs*i+0] 
        << " " << data[numOutputs*i+1]
        << " " << data[numOutputs*i+2]
        << " " << data[numOutputs*i+3]
        << " " << data[numOutputs*i+4]
        << " " << data[numOutputs*i+5]
        << endl;
    }

}

// Info<< "9" << nl << endl;

tf_utils::DeleteTensor(input_tensor_);
tf_utils::DeleteTensor(output_tensor_);
TF_DeleteSessionOptions(options_);
TF_DeleteStatus(status_);
tf_utils::DeleteSession(sess_);

// Info<< "10" << nl << endl;

auto endTauEqnTime = std::chrono::system_clock::now();
std::chrono::duration<double> TauEqnTime = endTauEqnTime - startTauEqnTime;

// Info << "TauEqn time taken: " << TauEqnTime << endl;
// std::cout << "elapsed time: " << TauEqnTime.count() << "s\n";

totalTauEqnTime = totalTauEqnTime + TauEqnTime.count();
double averageTauEqnTime = totalTauEqnTime / timeStep_n;

Info << "Average TauEqn time taken: " << averageTauEqnTime << endl;
