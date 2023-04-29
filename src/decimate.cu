#include "matx.h"

using namespace matx;

template<typename T>
__host__ T* decimate(T* data, int m, int n) {
    auto t = make_tensor<T>(data, {m, n});
}
