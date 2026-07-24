"""Shared test fixtures for cuda-scribe."""
import pytest


VECTOR_ADD_CUDA = """
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

REDUCTION_CUDA = """
__global__ void reduce_sum(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}
"""

SOFTMAX_CUDA = """
__global__ void softmax(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    // warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += expf(in_row[i] - max_val);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = expf(in_row[i] - max_val) / sum;
    }
}
"""


@pytest.fixture
def vector_add_cuda():
    return VECTOR_ADD_CUDA


@pytest.fixture
def reduction_cuda():
    return REDUCTION_CUDA


@pytest.fixture
def softmax_cuda():
    return SOFTMAX_CUDA
