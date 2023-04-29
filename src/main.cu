#include <driver_types.h>
#include <vector>
#include <iostream>
#include <H5Cpp.h>
#include "dtw.cu"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: No file path provided.\n";
        return 1;
    }
    H5::H5File file(argv[1], H5F_ACC_RDONLY);
    auto data_set = file.openDataSet("/data");

    auto data_space = data_set.getSpace();
    auto ndim = data_space.getSimpleExtentNdims();
    cout << "Dimension number: " << ndim << endl;
    vector<hsize_t> dims(ndim);
    data_space.getSimpleExtentDims(dims.data());
    cout << "Data Type: " << data_set.getTypeClass() << endl;
    cout << "Data Type size: " << data_set.getDataType().getSize() << endl;
    cout << "Data shape: " << dims[0] << " " << dims[1] << endl;
    vector<float> data(dims[0] * dims[1]);
    data_set.read(data.data(), {H5::PredType::NATIVE_FLOAT}, data_space, H5::DataSpace::ALL, {});
    float* res; cudaMalloc(&res, dims[1] * dims[1] * sizeof(float));
    float* D; cudaMalloc(&D, dims[0] * dims[1] * sizeof(float));
    cudaMemcpy(D, data.data(), dims[0] * dims[1] * sizeof(float), cudaMemcpyDefault);
    dtw(D, D, dims[1], dims[1], dims[0], dims[0], res);
}