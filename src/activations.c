#include "activations.h"

#include <math.h>
#include <stddef.h>

// Sigmoid activation function applied batch-wise in column-major format
void sigmoid_batch(const float *restrict input, float *restrict output,
                   const int batch_size, const int num_batches) {
  // Validate input parameters
  if (batch_size <= 0 || num_batches <= 0 || input == NULL || output == NULL)
    return;

  // Parallelize across batch elements and batches
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch_size; i++) {  // Iterate over elements in the batch
    for (int b = 0; b < num_batches; b++) {  // Iterate over batches
      float x = input[i * num_batches + b];
      float exp_neg = expf(-fabsf(x));

      // Apply the sigmoid function
      output[i * num_batches + b] =
          (x >= 0) ? 1.0f / (1.0f + exp_neg) : exp_neg / (1.0f + exp_neg);
    }
  }
}

// ReLU (Rectified Linear Unit) activation function applied batch-wise
void relu_batch(const float *restrict input, float *restrict output,
                const int batch_size, const int num_batches) {
  // Validate input parameters
  if (batch_size <= 0 || num_batches <= 0 || input == NULL || output == NULL)
    return;

  // Parallelize across batch elements and batches
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch_size; i++) {  // Iterate over elements in the batch
    for (int b = 0; b < num_batches; b++) {  // Iterate over batches
      float x = input[i * num_batches + b];

      // Apply the ReLU function
      output[i * num_batches + b] = (x >= 0) * x;
    }
  }
}

// Tanh activation function applied batch-wise in column-major format
void tanh_batch(const float *restrict input, float *restrict output,
                const int batch_size, const int num_batches) {
  // Validate input parameters
  if (batch_size <= 0 || num_batches <= 0 || input == NULL || output == NULL)
    return;

  // Parallelize across batch elements and batches
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch_size; i++) {  // Iterate over elements in the batch
    for (int b = 0; b < num_batches; b++) {  // Iterate over batches
      float x = input[i * num_batches + b];
      float exp_pos = expf(x);
      float exp_neg = expf(-x);

      // Apply the Tanh function
      output[i * num_batches + b] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
  }
}

// Leaky ReLU activation function applied batch-wise in column-major format
void leaky_relu_batch(const float *restrict input, float *restrict output,
                      const int batch_size, const int num_batches,
                      const float alpha) {
  // Validate input parameters
  if (batch_size <= 0 || num_batches <= 0 || input == NULL || output == NULL)
    return;

  // Parallelize across batch elements and batches
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch_size; i++) {  // Iterate over elements in the batch
    for (int b = 0; b < num_batches; b++) {  // Iterate over batches
      float x = input[i * num_batches + b];

      // Apply the Leaky ReLU function
      output[i * num_batches + b] = (x >= 0) ? x : alpha * x;
    }
  }
}

// Softmax function applied batch-wise in column-major format
void softmax_batch(const float *restrict input, float *restrict output,
                   const int batch_size, const int num_batches) {
  // Validate input parameters
  if (batch_size <= 0 || num_batches <= 0 || input == NULL || output == NULL)
    return;

  // Parallelize across batch elements
#pragma omp parallel for
  for (int i = 0; i < batch_size; i++) {
    // Find the maximum value for numerical stability
    float max_val = input[i * num_batches];
    for (int b = 1; b < num_batches; b++) {
      if (input[i * num_batches + b] > max_val)
        max_val = input[i * num_batches + b];
    }

    // Compute exponentials and their sum
    float sum = 0.0f;
    for (int b = 0; b < num_batches; b++) {
      output[i * num_batches + b] = expf(input[i * num_batches + b] - max_val);
      sum += output[i * num_batches + b];
    }

    // Normalize the output to ensure the probabilities sum to 1
    for (int b = 0; b < num_batches; b++) {
      output[i * num_batches + b] /= sum;
    }
  }
}
