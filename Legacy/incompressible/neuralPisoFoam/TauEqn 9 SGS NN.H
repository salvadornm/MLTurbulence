// Run the neural network to find the Tau field

volTensorField gradientUfull = fvc::grad(U);

const tensorField& gradientU = gradientUfull.internalField();
// gradientU = gradientUfull.internalField();
int numCells = mesh.cells().size(); //This works, returns expected value
for (int i = 0; i < numCells; i++)
{
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


Info<< "1" << nl << endl;

const int numInputs = 9;
const int numOutputs = 9;

const std::vector<std::int64_t> inputDims = {numCells,numInputs};

// Info<< "2" << nl << endl;

// volTensorField Tau = gradientUfull;

float gradOffset = -60;
float gradScale = 120;
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

    gradientUCount += 1;

}

// Info<< "3" << nl << endl;

auto input_tensor_ = tf_utils::CreateTensor(TF_FLOAT, inputDims, inputVals_);
TF_Tensor* output_tensor_ = nullptr;

// Info<< "4" << nl << endl;

auto graph_ = tf_utils::LoadGraph("/usr/include/NNTau003.pb");
if (graph_ == nullptr) {
    Info << "Can't load graph" << endl;
}

// Info<< "5" << nl << endl;

auto input_op = TF_Output{TF_GraphOperationByName(graph_,"dense_input"), 0};
auto out_op = TF_Output{TF_GraphOperationByName(graph_,"dense_2/BiasAdd"), 0};

// Info<< "6" << nl << endl;

auto status_ = TF_NewStatus();
auto options_ = TF_NewSessionOptions();
auto sess_ = TF_NewSession(graph_, options_, status_);

// Info<< "7" << nl << endl;

TF_SessionRun(sess_,
            nullptr, // Run options.
            &input_op, &input_tensor_, 1, // Input tensor ops, input tensor values, number of inputs.
            &out_op, &output_tensor_, 1, // Output tensor ops, output tensor values, number of outputs.
            nullptr, 0, // Target operations, number of targets.
            nullptr, // Run metadata.
            status_ // Output status.
            );

auto data = static_cast<float*>(TF_TensorData(output_tensor_));

// Info<< "8" << nl << endl;

float tauOffset = -30;
float tauScale = 60;
for (int i = 0; i < numCells; i++)
{
    Tau[i][0] = data[numOutputs*i+0]*tauScale + tauOffset;
    Tau[i][1] = data[numOutputs*i+1]*tauScale + tauOffset;
    Tau[i][2] = data[numOutputs*i+2]*tauScale + tauOffset;
    Tau[i][3] = data[numOutputs*i+3]*tauScale + tauOffset;
    Tau[i][4] = data[numOutputs*i+4]*tauScale + tauOffset;
    Tau[i][5] = data[numOutputs*i+5]*tauScale + tauOffset;
    Tau[i][6] = data[numOutputs*i+6]*tauScale + tauOffset;
    Tau[i][7] = data[numOutputs*i+7]*tauScale + tauOffset;
    Tau[i][8] = data[numOutputs*i+8]*tauScale + tauOffset;

}

// Info<< "9" << nl << endl;

tf_utils::DeleteTensor(input_tensor_);
tf_utils::DeleteTensor(output_tensor_);
TF_DeleteSessionOptions(options_);
TF_DeleteStatus(status_);
tf_utils::DeleteSession(sess_);

// Info<< "10" << nl << endl;
