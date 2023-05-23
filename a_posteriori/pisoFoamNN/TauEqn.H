// Run the neural network to find the Tau field
// Comments by Kacper & Alexander

volTensorField gradientUfull = fvc::grad(U);                        //create a tensor variable containing the gradient of the velocity field
const tensorField& gradientU = gradientUfull.internalField();       //create a pointer to the velocity gradient array
int numCells = mesh.cells().size();                                 //create a variable with thenumber of cells in the mesh

for (int i = 0; i < numCells; i++)                                  //loop through all cells and write the velocity graident in the variable to be saved in the simulation
{                                                                   //this variable servese no exact purpose but to verify the gradient against the residual stresses
     gradientUWrite[i][0] = gradientU[i][0];
     gradientUWrite[i][1] = gradientU[i][1];
     gradientUWrite[i][2] = gradientU[i][2];
     gradientUWrite[i][3] = gradientU[i][3];
     gradientUWrite[i][4] = gradientU[i][4];
     gradientUWrite[i][5] = gradientU[i][5];
     gradientUWrite[i][6] = gradientU[i][6];
     gradientUWrite[i][7] = gradientU[i][7];
     gradientUWrite[i][8] = gradientU[i][8];

     strainTensor[i][0] = gradientU[i][0];
     strainTensor[i][1] = 1/2 * (gradientU[i][1] + gradientU[i][3]);
     strainTensor[i][2] = 1/2 * (gradientU[i][2] + gradientU[i][6]);
     strainTensor[i][3] = 1/2 * (gradientU[i][1] + gradientU[i][3]);
     strainTensor[i][4] = gradientU[i][4];
     strainTensor[i][5] = 1/2 * (gradientU[i][5] + gradientU[i][7]);
     strainTensor[i][6] = 1/2 * (gradientU[i][2] + gradientU[i][6]);
     strainTensor[i][7] = 1/2 * (gradientU[i][5] + gradientU[i][7]);
     strainTensor[i][8] = gradientU[i][8];
}

const tensorField& strain = strainTensor.internalField();

volVectorField Ufull = U;
const vectorField& Uval = Ufull.internalField();

// for (int i = 0; i < numCells; i++)                                  //loop through all cells and write the velocity graident in the variable to be saved in the simulation
// {                                                                   //this variable servese no exact purpose but to verify the gradient against the residual stresses
//      UWrite[i][0] = Uval[i][0];
//      UWrite[i][1] = Uval[i][1];
//      UWrite[i][2] = Uval[i][2];
// }



// Info<< "1" << nl << endl;

const int numInputs = 18;            //define the number of inputs to the network
const int numOutputs = 6;           //define the number of outputs to to the network

const std::vector<std::int64_t> inputDims = {numCells,numInputs};       //create a vector variable with the dimensions: number of cells and number of inputs

// Info<< "2" << nl << endl;



//define a vector with the offset parameters to the input
float gradOffset[9] = {
    -3.900756817311048508e+00,
    -1.444765054324473681e+02,
    -1.137467156723141670e+01,
    -2.693360943347215652e+00,
    -5.753553645685315132e+00,
    -7.548901014961302280e+00,
    -3.896191398613154888e+00,
    -3.759729907005566929e+01,
    -4.985060887993313372e+00
};


//define a vector with the scaling parameters to the input
float gradScale[9] = {
    7.016355128958821297e+00,
    2.702295231262053790e+02,
    2.210854010656476021e+01,
    5.251506369328126311e+00,
    1.052136283571599051e+01,
    1.364857028890401125e+01,
    7.137969396484550089e+00,
    8.018603350436205801e+01,
    1.038676211016718298e+01
};


float velOffset[3] = {
    3.718057181686162949e-03,
    -1.677487641572952271e-01,
    -2.791506052017211914e-01
};


float velScale[3] = {
    1.240572056267410517e+00,
    3.537947982549667358e-01,
    5.110933333635330200e-01
};



float strainOffset[6] = {
    -3.900756817311048508e+00,
    -5.753553645685315132e+00,
    -4.985060887993313372e+00,
    -7.223845219484218205e+01,
    -5.366458723961841315e+00,
    -1.880007189657938582e+01
};


float strainScale[6] = {
    7.016355128958821297e+00,
    1.052136283571599051e+01,
    1.038676211016718298e+01,
    1.351152864862106355e+02,
    1.025323783433850622e+01,
    4.009687982726809707e+01
};


int gradientUCount = 0;             //define a counter and set it to 0
std::vector<float> inputVals_;      //create a vector 'inputVals_'
forAll(gradientU,cellI)             //loop through all cells
{
    const tensor L = gradientU[cellI];                      //create a tensor L containing all values of gradientU for one cell
    const vector M = Uval[cellI];
    const tensor N = strain[cellI];

    float g0 = (M.x() - velOffset[0]) / velScale[0];
    float g1 = (M.y() - velOffset[1]) / velScale[1];
    float g2 = (M.z() - velOffset[2]) / velScale[2];

    //scale the gradients
    float g3 = (L.xx() - gradOffset[0]) / gradScale[0];
    float g4 = (L.xy() - gradOffset[1]) / gradScale[1];
    float g5 = (L.xz() - gradOffset[2]) / gradScale[2];
    float g6 = (L.yx() - gradOffset[3]) / gradScale[3];
    float g7 = (L.yy() - gradOffset[4]) / gradScale[4];
    float g8 = (L.yz() - gradOffset[5]) / gradScale[5];
    float g9 = (L.zx() - gradOffset[6]) / gradScale[6];
    float g10 = (L.zy() - gradOffset[7]) / gradScale[7];
    float g11 = (L.zz() - gradOffset[8]) / gradScale[8];

    float g12 = (N.xx() - strainOffset[0]) / strainScale[0];
    float g13 = (N.yy() - strainOffset[1]) / strainScale[1];
    float g14 = (N.zz() - strainOffset[2]) / strainScale[2];
    float g15 = (N.xy() - strainOffset[3]) / strainScale[3];
    float g16 = (N.xz() - strainOffset[4]) / strainScale[4];
    float g17 = (N.yz() - strainOffset[5]) / strainScale[5];


    //push the scaled gradients back into the 'inputVals_' vector
    inputVals_.push_back(g0);
    inputVals_.push_back(g1);
    inputVals_.push_back(g2);
    inputVals_.push_back(g3);
    inputVals_.push_back(g4);
    inputVals_.push_back(g5);
    inputVals_.push_back(g6);
    inputVals_.push_back(g7);
    inputVals_.push_back(g8);
    inputVals_.push_back(g9);
    inputVals_.push_back(g10);
    inputVals_.push_back(g11);
    inputVals_.push_back(g12);
    inputVals_.push_back(g13);
    inputVals_.push_back(g14);
    inputVals_.push_back(g15);
    inputVals_.push_back(g16);
    inputVals_.push_back(g17);

    gradientUCount += 1;            //increment the counter
}

// Info<< "3" << nl << endl;

auto input_tensor_ = tf_utils::CreateTensor(TF_FLOAT, inputDims, inputVals_);       //create a tensorflow tensor of type float of dimension inputDims and containig inputVals_
TF_Tensor* output_tensor_ = nullptr;                                                //crreate a null pointer to the output tensor

// Info<< "4" << nl << endl;

auto graph_ = tf_utils::LoadGraph("./model_velocity_gradient_strain.pb");                                //load the network graph

//send an error message if graph cannot be loaded
if (graph_ == nullptr) {
    Info << "Can't load graph" << endl;
}

// Info<< "5" << nl << endl;

auto input_op = TF_Output{TF_GraphOperationByName(graph_,"dense_input"), 0};            //set the name of the input layer
auto out_op = TF_Output{TF_GraphOperationByName(graph_,"dense_3/BiasAdd"), 0};          //set the name of the output layer to the tensorflow denomination of the output layer

// Info<< "6" << nl << endl;

//set up the new session
auto status_ = TF_NewStatus();
auto options_ = TF_NewSessionOptions();
auto sess_ = TF_NewSession(graph_, options_, status_);

// Info<< "7" << nl << endl;

//run the tensorflow session with the prescribed input
TF_SessionRun(sess_,
            nullptr, // Run options.
            &input_op, &input_tensor_, 1, // Input tensor ops, input tensor values, number of inputs.
            &out_op, &output_tensor_, 1, // Output tensor ops, output tensor values, number of outputs.
            nullptr, 0, // Target operations, number of targets.
            nullptr, // Run metadata.
            status_ // Output status.
            );

auto data = static_cast<float*>(TF_TensorData(output_tensor_));                         //retrieve the output of the tensorflow network

// Info<< "8" << nl << endl;


//define a vector with the offset parameters to the output
float tauOffset[6] = {
    5.688426512051591999e-07,
    8.068402993676310146e-11,
    2.213350414497093325e-07,
    -1.277603468991839186e-02,
    -1.543771831029694397e-02,
    -8.130325771895041687e-03
};


//define a vector with the scaling parameters to the output
float tauScale[6] = {
    5.937449358660185084e-02,
    2.143822076922395625e-02,
    2.301486297504369213e-02,
    3.263759854756043133e-02,
    3.426206730031200465e-02,
    1.603513716256727006e-02
};


for (int i = 0; i < numCells; i++)                                  //loop through all cells
{
    //scale all outputs accordingly
    TauNN[i][0] = data[numOutputs*i+0]*tauScale[0] + tauOffset[0];  //xx
    TauNN[i][1] = data[numOutputs*i+3]*tauScale[3] + tauOffset[3];  //xy
    TauNN[i][2] = data[numOutputs*i+4]*tauScale[4] + tauOffset[4];  //xz
    TauNN[i][3] = data[numOutputs*i+3]*tauScale[3] + tauOffset[3];  //xy
    TauNN[i][4] = data[numOutputs*i+1]*tauScale[1] + tauOffset[1];  //yy
    TauNN[i][5] = data[numOutputs*i+5]*tauScale[5] + tauOffset[5];  //yz
    TauNN[i][6] = data[numOutputs*i+4]*tauScale[4] + tauOffset[4];  //xz
    TauNN[i][7] = data[numOutputs*i+5]*tauScale[5] + tauOffset[5];  //yz
    TauNN[i][8] = data[numOutputs*i+2]*tauScale[2] + tauOffset[2];  //zz

}

// Info<< "9" << nl << endl;

tf_utils::DeleteTensor(input_tensor_);                              //delete the input tensor
tf_utils::DeleteTensor(output_tensor_);                             //delete the output tensor

//delete the tensorflow session
TF_DeleteSessionOptions(options_);
TF_DeleteStatus(status_);
tf_utils::DeleteSession(sess_);

// Info<< "10" << nl << endl;


// TauNN.correctBoundaryConditions();