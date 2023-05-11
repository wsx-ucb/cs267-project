#include <cmath>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

using namespace std;

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
    return abs(a - b);
}

template<int k, typename T>
__device__ void dtw_lane(T* Q, T* S, T* Mv, int Ql, int Sl, int w, int t) {
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
    for (int i = 0; i < Ql + t - 2; i++) {
        int li = i - threadIdx.x;
        if (i % WARP_SIZE == 0 && i + 1 + threadIdx.x < Ql) {
            Qn = Q[i + 1 + threadIdx.x];
        }
        Qc = __shfl_up_sync(0xffffffff, Qc, 1);
        if (threadIdx.x == 0) Qc = Qn;
        Qn = __shfl_down_sync(0xffffffff, Qn, 1);
        M_ul = M_l;
        M_l = __shfl_up_sync(0xffffffff, lM[w - 1], 1);
        if (threadIdx.x == 0 && li + 1 < Ql) M_l = Mv[li + 1];
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
__global__ void dtw_kernel(T* Qs, T* Ss, int Ql, int Sl, T* dist, bool diag) {
    __shared__ T* Mv;
    if (diag && blockIdx.x < blockIdx.y) return;
    if (threadIdx.x == 0) {
        Mv = new T[Ql];
        for (int i = 0; i < Ql; i++) Mv[i] = INFINITY;
        Mv[0] = 0;
    }
    int j = 1;
    for (; j + k * WARP_SIZE < Sl; j += k * WARP_SIZE) {
        dtw_lane<k, T>(Qs + blockIdx.x * Ql, Ss + blockIdx.y * Sl + j, Mv, Ql, Sl, k, WARP_SIZE);
    }
    // Edge cases
    dtw_lane<k, T>(Qs + blockIdx.x * Ql, Ss + blockIdx.y * Sl + j, Mv, Ql, Sl, (Sl - j) / WARP_SIZE, WARP_SIZE);
    j += (Sl - j) / WARP_SIZE * WARP_SIZE;
    dtw_lane<k, T>(Qs + blockIdx.x * Ql, Ss + blockIdx.y * Sl + j, Mv, Ql, Sl, 1, Sl - j);
    __syncthreads();
    if (threadIdx.x == 0) {
        dist[blockIdx.x * gridDim.y + blockIdx.y] = Mv[Ql - 1];
        if (diag) dist[blockIdx.y * gridDim.y + blockIdx.x] = Mv[Ql - 1];
        delete[] Mv;
    }
}

int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// l - length of one channel
// n - number of channels
template<typename T>
__host__ void dtw(T* Cs, int l, int n, T* res) {
    int dev_count; CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int s = div_up(n, dev_count);
    struct task {
        T* Q;
        T* S;
        T* D;
        int Qn;
        int Sn;
        cudaStream_t s;
    };
    CUDA_CHECK(cudaSetDevice(0));
    T* D; CUDA_CHECK(cudaMalloc(&D, n * n * sizeof(T)));
    vector<task> tasks(dev_count);
    for (int i = 0; i < dev_count; i++) {
        auto& t = tasks[i];
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, l * s * s * sizeof(T)));
        CUDA_CHECK(cudaStreamCreate(&t.s));
        for (int j = 0; j < dev_count; j++) {
            if (i != j) CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
        }
        CUDA_CHECK(cudaMallocAsync(&t.Q, s * l * sizeof(T), t.s));
        t.Qn = min(s, n - i * s);
        CUDA_CHECK(cudaMemcpyAsync(t.Q, Cs + i * s * l, t.Qn * l * sizeof(T), cudaMemcpyDefault, t.s));
        CUDA_CHECK(cudaMallocAsync(&t.S, s * l * sizeof(T), t.s));
        CUDA_CHECK(cudaMallocAsync(&t.D, s * s * sizeof(T), t.s));
    }
    // Run DTW
    for (int j = 0; j < div_up(dev_count + 1, 2); j++) {
        for (int i = 0; i < dev_count; i++) {
            int x = i * s;
            int y = ((i + j) % dev_count) * s;
            // Prepare the task
            auto& t = tasks[i];
            CUDA_CHECK(cudaSetDevice(i));
            int peer = (i + j) % dev_count;
            CUDA_CHECK(cudaMemcpyPeerAsync(t.S, i, tasks[peer].Q, peer, tasks[peer].Qn * l * sizeof(T), t.s));
            t.Sn = min(s, n - y);
            // Run the task
            dim3 blocks(t.Qn, t.Sn);
            dtw_kernel<32, T><<<blocks, WARP_SIZE, 0, t.s>>>(t.Q, t.S, l, l, t.D, x == y);
            // Collect result
            for (int k = 0; k < t.Qn; k++) {
                CUDA_CHECK(cudaMemcpyAsync(D + (x + k) * n + y, t.D + k * t.Sn, t.Sn * sizeof(T), cudaMemcpyDefault, t.s));
            }
        }
    }
    for (int i = 0; i < dev_count; i++) {
        auto& t = tasks[i];
        CUDA_CHECK(cudaStreamSynchronize(t.s));
        CUDA_CHECK(cudaStreamDestroy(t.s));
        CUDA_CHECK(cudaFree(t.Q));
        CUDA_CHECK(cudaFree(t.S));
        CUDA_CHECK(cudaFree(t.D));
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(res, D, n * n * sizeof(T), cudaMemcpyDefault));
    CUDA_CHECK(cudaFree(D));
    // Fill in by symmetry
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (res[i * n + j] == 0) {
                res[i * n + j] = res[j * n + i];
            } else if (res[j * n + i] == 0) {
                res[j * n + i] == res[i * n + j];
            }
        }
    }
}
