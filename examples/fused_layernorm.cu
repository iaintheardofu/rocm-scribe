// Example 5: Fused Layer Normalization — Tests multi-pass fusion
// Expected: Tests operator fusion capability and mixed-precision handling
__global__ void fused_layernorm(
    const float *input, const float *weight, const float *bias,
    float *output,
    int batch_size, int hidden_size, float eps
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float *row_in = input + row * hidden_size;
    float *row_out = output + row * hidden_size;

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        mean += row_in[i];
    }
    mean /= hidden_size;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float diff = row_in[i] - mean;
        var += diff * diff;
    }
    var /= hidden_size;

    // Normalize, scale, and shift
    float inv_std = rsqrtf(var + eps);
    for (int i = 0; i < hidden_size; i++) {
        row_out[i] = (row_in[i] - mean) * inv_std * weight[i] + bias[i];
    }
}
