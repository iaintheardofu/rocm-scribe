// Example 4: Online Softmax — Tests numerical stability translation
// Expected: LLM may miss max-subtraction stability trick
__global__ void softmax_forward(
    const float *input, float *output,
    int batch_size, int num_classes
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float *row_input = input + row * num_classes;
    float *row_output = output + row * num_classes;

    // Pass 1: Find max for numerical stability
    float max_val = -1e20f;
    for (int i = 0; i < num_classes; i++) {
        max_val = fmaxf(max_val, row_input[i]);
    }

    // Pass 2: Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        float val = expf(row_input[i] - max_val);
        row_output[i] = val;
        sum += val;
    }

    // Pass 3: Normalize
    for (int i = 0; i < num_classes; i++) {
        row_output[i] /= sum;
    }
}
