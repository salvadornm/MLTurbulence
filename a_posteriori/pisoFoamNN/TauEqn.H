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
}



// Info<< "1" << nl << endl;

const int numInputs = 9;            //define the number of inputs to the network
const int numOutputs = 6;           //define the number of outputs to to the network

const std::vector<std::int64_t> inputDims = {numCells,numInputs};       //create a vector variable with the dimensions: number of cells and number of inputs

// Info<< "2" << nl << endl;



//define a vector with the offset parameters to the input
float gradOffset[9] = {             
    -4.958539862185716629e+00,
    -8.040475547313690186e+00,
    -1.138027196114126127e+01,
    -3.540999109158292413e+00,
    -6.256312112789601088e+00,
    -7.548901014961302280e+00,
    -3.896191398613154888e+00,
    -5.114444367692340165e+01,
    -5.097333902958780527e+00
};


//define a vector with the scaling parameters to the input
float gradScale[9] = {              
    8.089620698243379593e+00,
    1.475831411480903625e+02,
    2.242399184182431782e+01,
    6.368195477407425642e+00,
    1.102412130282027647e+01,
    1.476183353085070848e+01,
    7.678800913039594889e+00,
    9.373317811122979037e+01,
    1.089706473471596837e+01
};


int gradientUCount = 0;             //define a counter and set it to 0
std::vector<float> inputVals_;      //create a vector 'inputVals_'
forAll(gradientU,cellI)             //loop through all cells
{
    const tensor L = gradientU[cellI];                      //create a tensor L containing all values of gradientU for one cell

    //scale the gradients
    float g0 = (L.xx() - gradOffset[0]) / gradScale[0];
    float g1 = (L.xy() - gradOffset[1]) / gradScale[1];
    float g2 = (L.xz() - gradOffset[2]) / gradScale[2];
    float g3 = (L.yx() - gradOffset[3]) / gradScale[3];
    float g4 = (L.yy() - gradOffset[4]) / gradScale[4];
    float g5 = (L.yz() - gradOffset[5]) / gradScale[5];
    float g6 = (L.zx() - gradOffset[6]) / gradScale[6];
    float g7 = (L.zy() - gradOffset[7]) / gradScale[7];
    float g8 = (L.zz() - gradOffset[8]) / gradScale[8];

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

    gradientUCount += 1;            //increment the counter
}

// Info<< "3" << nl << endl;

auto input_tensor_ = tf_utils::CreateTensor(TF_FLOAT, inputDims, inputVals_);       //create a tensorflow tensor of type float of dimension inputDims and containig inputVals_
TF_Tensor* output_tensor_ = nullptr;                                                //crreate a null pointer to the output tensor

// Info<< "4" << nl << endl;

auto graph_ = tf_utils::LoadGraph("./model_new.pb");                                //load the network graph

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
    6.858863343950361013e-07,   //xx
    1.228890507934036447e-11,   //yy
    5.800112038262230815e-08,   //zz
    -1.638958986677607754e-02,  //xy
    -1.509368470797323170e-02,  //xz
    -1.048374631993453931e-02   //yz
};


//define a vector with the scaling parameters to the output
float tauScale[6] = {
    5.857573601825549758e-02,   //xx
    2.550598112970951792e-02,   //yy
    2.596298144001103411e-02,   //zz
    2.231063950052669664e-02,   //xy
    2.952483214836743031e-02,   //xz
    1.941745822061179128e-02    //yz
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