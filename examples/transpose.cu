// Example 2: Shared Memory Transpose — Tests bank conflict handling
// Expected: LLM produces naive version, needs shared memory swizzle fix
__global__ void transpose(const float *input, float *output, int M, int N) {
    __shared__ float tile[32][33];  // +1 padding to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < N && y < M)
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];

    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < M && y < N)
        output[y * M + x] = tile[threadIdx.x][threadIdx.y];
}
