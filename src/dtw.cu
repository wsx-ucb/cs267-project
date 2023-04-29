#include <cmath>
#include <iostream>

#define WARP_SIZE 32

#define CUDA_CHECK(status) \
  do { \
    cudaError_t _status = (status); \
    if (_status != cudaSuccess) { \
      std::cerr << "CUDA error: " << cudaGetErrorString(_status) << std::endl; \
      exit(1); \
    } \
  } while (false)

template<typename T>
__device__ T d(T a, T b) {
    return (a - b) * (a - b);
}

template<int k, typename T>
__device__ void dtw_lane(T* Q, T* S, T* Mv, int m, int n, int w, int t) {
    // w <= k, t <= WARP_SIZE
    T lS[k], lM[k];
    T Qc, Qn;
    T M_ul, M_l = Mv[0];
    __syncthreads();
    for (auto& lm : lM) lm = INFINITY;
    if (threadIdx.x < t) {
        for (int l = 0; l < w; l++) {
            lS[l] = S[threadIdx.x * w + l];
        }
    }
    for (int i = 0; i < m + t - 2; i++) {
        int li = i - threadIdx.x;
        if (i % WARP_SIZE == 0 && i + 1 + threadIdx.x < m) {
            Qn = Q[i + 1 + threadIdx.x];
        }
        Qc = __shfl_up_sync(0xffffffff, Qc, 1);
        if (threadIdx.x == 0) Qc = Qn;
        Qn = __shfl_down_sync(0xffffffff, Qn, 1);
        M_ul = M_l;
        M_l = __shfl_up_sync(0xffffffff, lM[w - 1], 1);
        if (threadIdx.x == 0 && li + 1 < m) M_l = Mv[li + 1];
        T M_left = M_l, M_upleft = M_ul;
        for (int l = 0; l < w && li >= 0; l++) {
            T M_new = d(Qc, lS[l]) + min(M_left, min(lM[l], M_upleft));
            M_upleft = lM[l];
            lM[l] = M_left = M_new;
        }
        if (threadIdx.x == t - 1 && li >= 0) Mv[li + 1] = lM[w - 1];
    }
    if (threadIdx.x == t - 1) Mv[0] = INFINITY;
}

// Call with 32 threads in a block,
// And each block for a (Q, S) pair.
template<int k, typename T>
__global__ void dtw_kernel(T* Qs, T* Ss, int m, int n, T* dist) {
    __shared__ T* Mv;
    if (blockIdx.x < blockIdx.y) return;
    if (threadIdx.x == 0) {
        Mv = new T[m];
        for (int i = 0; i < m; i++) Mv[i] = INFINITY;
        Mv[0] = 0;
    }
    int j = 1;
    for (; j + k * WARP_SIZE < n; j += k * WARP_SIZE) {
        dtw_lane<k, T>(Qs + blockIdx.x * m, Ss + blockIdx.y * n + j, Mv, m, n, k, WARP_SIZE);
    }
    // Edge cases
    dtw_lane<k, T>(Qs + blockIdx.x * m, Ss + blockIdx.y * n + j, Mv, m, n, (n - j) / WARP_SIZE, WARP_SIZE);
    j += (n - j) / WARP_SIZE * WARP_SIZE;
    dtw_lane<k, T>(Qs + blockIdx.x * m, Ss + blockIdx.y * n + j, Mv, m, n, 1, n - j);
    __syncthreads();
    if (threadIdx.x == 0) {
        dist[blockIdx.x * gridDim.y + blockIdx.y] = Mv[m - 1];
        dist[blockIdx.y * gridDim.y + blockIdx.x] = Mv[m - 1];
        delete[] Mv;
    }
}

template<typename T>
__host__ void dtw(T* Qs, T* Ss, int m, int n, int qn, int sn, T* res) {
    size_t mem; CUDA_CHECK(cudaDeviceGetLimit(&mem, cudaLimitMallocHeapSize));
    printf("Current heap size %zu\n", mem);
    printf("Setting heap size %zu\n", m * sn * qn * sizeof(T));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, m * sn * qn * sizeof(
    T)));
    CUDA_CHECK(cudaDeviceGetLimit(&mem, cudaLimitMallocHeapSize));
    printf("Current heap size %zu\n", mem);
    dim3 blocks(qn, sn);
    dtw_kernel<32, T><<<blocks, WARP_SIZE>>>(Qs, Ss, m, n, res);
    CUDA_CHECK(cudaDeviceSynchronize());
    T* r = new T[qn * sn];
    CUDA_CHECK(cudaMemcpy(r, res, qn * sn * sizeof(T), cudaMemcpyDefault));
    for (int i = 0; i < qn; i++) {
        for (int j = 0; j < sn; j++) {
            printf("%f\t", r[i * sn + j]);
        }
        printf("\n");
    }
    delete[] r;
}
