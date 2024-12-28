#include "test_activations.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void test_sigmoid_batch(void) {
  float input[] = {0, 10, -10, 1, -1, 5};  // Column-major order
  float output[6] = {0};
  sigmoid_batch(input, output, 3, 2);  // batch_size = 3, num_batches = 2

  assert(fabsf(output[0] - 0.5f) < 1e-6f);
  assert(fabsf(output[1] - 0.9999546f) < 1e-6f);
  assert(fabsf(output[2] - 0.0000454f) < 1e-6f);
  assert(fabsf(output[3] - 0.7310586f) < 1e-6f);
  assert(fabsf(output[4] - 0.2689414f) < 1e-6f);
  assert(fabsf(output[5] - 0.9933071f) < 1e-6f);
}

static void test_relu_batch(void) {
  float input[] = {-5.0f, 0.0f, 5.0f,
                   -1.0f, 2.0f, -3.0f};  // Column-major order
  float output[6] = {0};
  relu_batch(input, output, 3, 2);  // batch_size = 3, num_batches = 2

  assert(output[0] == 0.0f);
  assert(output[1] == 0.0f);
  assert(output[2] == 5.0f);
  assert(output[3] == 0.0f);
  assert(output[4] == 2.0f);
  assert(output[5] == 0.0f);
}

static void test_tanh_batch(void) {
  float input[] = {0, 1, -1, 5, -5, 10};  // Column-major order
  float output[6] = {0};
  tanh_batch(input, output, 3, 2);  // batch_size = 3, num_batches = 2

  assert(fabsf(output[0] - 0.0f) < 1e-6f);
  assert(fabsf(output[1] - 0.7615942f) < 1e-6f);
  assert(fabsf(output[2] + 0.7615942f) < 1e-6f);
  assert(fabsf(output[3] - 0.9999092f) < 1e-6f);
  assert(fabsf(output[4] + 0.9999092f) < 1e-6f);
  assert(fabsf(output[5] - 1.0f) < 1e-6f);
}

static void test_leaky_relu_batch(void) {
  float input[] = {-5.0f, 0.0f, 5.0f,
                   -1.0f, 2.0f, -3.0f};  // Column-major order
  float output[6] = {0};
  float alpha = 0.1f;
  leaky_relu_batch(input, output, 3, 2,
                   alpha);  // batch_size = 3, num_batches = 2

  assert(fabsf(output[0] - (-5.0f * alpha)) < 1e-6f);
  assert(output[1] == 0.0f);
  assert(output[2] == 5.0f);
  assert(fabsf(output[3] - (-1.0f * alpha)) < 1e-6f);
  assert(output[4] == 2.0f);
  assert(fabsf(output[5] - (-3.0f * alpha)) < 1e-6f);
}

static void test_softmax_batch(void) {
  float input[] = {1, 2, 3, 1, 2, 3};  // Column-major order
  float output[6] = {0};
  softmax_batch(input, output, 3, 2);  // batch_size = 3, num_batches = 2

  // Check normalization for each batch (column-major indexing)
  for (int i = 0; i < 3; i++) {
    float sum = 0.0f;
    for (int b = 0; b < 2; b++) {
      sum += output[i * 2 + b];  // Adjust index to column-major
    }
    assert(fabsf(sum - 1.0f) < 1e-6f);
  }

  // Validate individual softmax values (column-major indexing)
}

void test_activations(void) {
  test_sigmoid_batch();
  test_relu_batch();
  test_tanh_batch();
  test_leaky_relu_batch();
  test_softmax_batch();

  printf("All activation tests passed!\n");
}
