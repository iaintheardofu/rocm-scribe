// Example 3: Parallel Reduction — Tests determinism across backends
// Expected: LLM uses tl.sum (non-deterministic); must verify ULP tolerance
__global__ void reduce_sum(const float *input, float *output, int N) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset)
            shared[tid] += shared[tid + offset];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = shared[0];
}
