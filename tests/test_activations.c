#include "test_activations.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void test_sigmoid_batch(void) {
  // Input in column-major order: batch_size = 3, input_size = 2
  float input[] = {0.0f, 10.0f, -10.0f, 1.0f, -1.0f, 5.0f};
  float output[6] = {0};

  sigmoid_batch(input, output, 3, 2);

  // Compare results with expected sigmoid values
  // Index mapping: output[idx] = output[feature * batch_size + sample]
  // (feature = 0..1, sample = 0..2)
  assert(fabsf(output[0] - 0.5f) < 1e-6f);        // sigmoid(0)
  assert(fabsf(output[1] - 0.9999546f) < 1e-6f);  // sigmoid(10)
  assert(fabsf(output[2] - 0.0000454f) < 1e-6f);  // sigmoid(-10)
  assert(fabsf(output[3] - 0.7310586f) < 1e-6f);  // sigmoid(1)
  assert(fabsf(output[4] - 0.2689414f) < 1e-6f);  // sigmoid(-1)
  assert(fabsf(output[5] - 0.9933071f) < 1e-6f);  // sigmoid(5)
}

static void test_relu_batch(void) {
  // Input in column-major order: batch_size = 3, input_size = 2
  float input[] = {-5.0f, 0.0f, 5.0f, -1.0f, 2.0f, -3.0f};
  float output[6] = {0};

  relu_batch(input, output, 3, 2);

  // Compare ReLU outputs
  // For ReLU, negative values become 0
  assert(output[0] == 0.0f);  // relu(-5)
  assert(output[1] == 0.0f);  // relu(0)
  assert(output[2] == 5.0f);  // relu(5)
  assert(output[3] == 0.0f);  // relu(-1)
  assert(output[4] == 2.0f);  // relu(2)
  assert(output[5] == 0.0f);  // relu(-3)
}

static void test_tanh_batch(void) {
  // Input in column-major order: batch_size = 3, input_size = 2
  float input[] = {0.0f, 1.0f, -1.0f, 5.0f, -5.0f, 10.0f};
  float output[6] = {0};

  tanh_batch(input, output, 3, 2);

  // Compare with expected tanh values
  assert(fabsf(output[0] - 0.0f) < 1e-6f);        // tanh(0)
  assert(fabsf(output[1] - 0.7615942f) < 1e-6f);  // tanh(1)
  assert(fabsf(output[2] + 0.7615942f) < 1e-6f);  // tanh(-1)
  assert(fabsf(output[3] - 0.9999092f) < 1e-6f);  // tanh(5) ~ 0.9999092
  assert(fabsf(output[4] + 0.9999092f) < 1e-6f);  // tanh(-5) ~ -0.9999092
  assert(fabsf(output[5] - 1.0f) < 1e-6f);  // tanh(10) ~ 0.9999999 (approx 1.0)
}

static void test_leaky_relu_batch(void) {
  // Input in column-major order: batch_size = 3, input_size = 2
  float input[] = {-5.0f, 0.0f, 5.0f, -1.0f, 2.0f, -3.0f};
  float output[6] = {0};
  float alpha = 0.1f;

  leaky_relu_batch(input, output, 3, 2, alpha);

  // Compare with expected Leaky ReLU outputs
  assert(fabsf(output[0] - (-5.0f * alpha)) < 1e-6f);
  assert(output[1] == 0.0f);
  assert(output[2] == 5.0f);
  assert(fabsf(output[3] - (-1.0f * alpha)) < 1e-6f);
  assert(output[4] == 2.0f);
  assert(fabsf(output[5] - (-3.0f * alpha)) < 1e-6f);
}

static void test_softmax_batch(void) {
  // Input in column-major order: batch_size = 3, input_size = 2
  float input[] = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
  float output[6] = {0};

  softmax_batch(input, output, 3, 2);

  // Verify that each column sums to 1
  // Column-major indexing: output[feature * batch_size + sample]
  for (int row = 0; row < 3; row++) {
    float sum = 0.0f;
    for (int col = 0; col < 2; col++) {
      sum += output[row * 2 + col];
    }
    assert(fabsf(sum - 1.0f) < 1e-6f);
  }
}

void test_activations(void) {
  test_sigmoid_batch();
  test_relu_batch();
  test_tanh_batch();
  test_leaky_relu_batch();
  test_softmax_batch();

  printf("All activation tests passed!\n");
}
