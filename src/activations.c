#include "activations.h"

// Sigmoid activation function applied column-wise
void sigmoid_batch(const float* input, float* output, const int batch_size,
                   const int input_size) {
  if (batch_size <= 0 || input_size <= 0 || input == NULL || output == NULL)
    return;

#pragma omp parallel for collapse(2)
  for (int feature = 0; feature < input_size; feature++) {
    for (int sample = 0; sample < batch_size; sample++) {
      int idx = feature * batch_size + sample;
      float x = input[idx];
      float exp_neg = expf(-fabsf(x));

      output[idx] =
          (x >= 0) ? 1.0f / (1.0f + exp_neg) : exp_neg / (1.0f + exp_neg);
    }
  }
}

// ReLU activation function applied column-wise
void relu_batch(const float* input, float* output, const int batch_size,
                const int input_size) {
  if (batch_size <= 0 || input_size <= 0 || input == NULL || output == NULL)
    return;

#pragma omp parallel for collapse(2)
  for (int feature = 0; feature < input_size; feature++) {
    for (int sample = 0; sample < batch_size; sample++) {
      int idx = feature * batch_size + sample;
      float x = input[idx];
      output[idx] = (x >= 0) ? x : 0.0f;
    }
  }
}

// Tanh activation function applied column-wise
void tanh_batch(const float* input, float* output, const int batch_size,
                const int input_size) {
  if (batch_size <= 0 || input_size <= 0 || input == NULL || output == NULL)
    return;

#pragma omp parallel for collapse(2)
  for (int feature = 0; feature < input_size; feature++) {
    for (int sample = 0; sample < batch_size; sample++) {
      int idx = feature * batch_size + sample;
      float x = input[idx];
      float exp_pos = expf(x);
      float exp_neg = expf(-x);

      output[idx] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
  }
}

// Leaky ReLU activation function applied column-wise
void leaky_relu_batch(const float* input, float* output, const int batch_size,
                      const int input_size, const float alpha) {
  if (batch_size <= 0 || input_size <= 0 || input == NULL || output == NULL)
    return;

#pragma omp parallel for collapse(2)
  for (int feature = 0; feature < input_size; feature++) {
    for (int sample = 0; sample < batch_size; sample++) {
      int idx = feature * batch_size + sample;
      float x = input[idx];
      output[idx] = (x >= 0) ? x : alpha * x;
    }
  }
}

// Softmax function applied column-wise
void softmax_batch(const float* input, float* output, const int batch_size,
                   const int input_size) {
  if (batch_size <= 0 || input_size <= 0 || input == NULL || output == NULL)
    return;

#pragma omp parallel for
  for (int sample = 0; sample < batch_size; sample++) {
    float max_val = input[sample];
    for (int feature = 1; feature < input_size; feature++) {
      int idx = feature * batch_size + sample;
      if (input[idx] > max_val) max_val = input[idx];
    }

    // Compute softmax in one pass
    float sum = 0.0f;
    for (int feature = 0; feature < input_size; feature++) {
      int idx = feature * batch_size + sample;
      output[idx] = expf(input[idx] - max_val);
      sum += output[idx];
    }

    // Normalize the output
    for (int feature = 0; feature < input_size; feature++) {
      int idx = feature * batch_size + sample;
      output[idx] /= sum;
    }
  }
}
