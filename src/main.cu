#include <vector>
#include <iostream>
#include <chrono>
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
    const char* channel_num = getenv("DAS_CHANNEL_NUM");
    int n = min(channel_num == NULL ? INT_MAX : atoi(channel_num), (int) dims[0]);
    vector<float> res(n * n);
    auto start = std::chrono::high_resolution_clock::now();
    dtw(data.data(), dims[1], n, res.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    printf("Performed DTW in %f s\n", duration);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f\t", res[i * n + j]);
        }
        printf("\n");
    }
}