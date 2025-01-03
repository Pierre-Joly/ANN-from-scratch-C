#include "test_loss.h"

/**
 * Tests for MSE_forward and MSE_backward
 */
static void test_MSE(void) {
  // Test data
  const int batch_size = 2;
  const int num_features = 3;

  // prediction and ground_truth
  // Indexing format: prediction[feature * batch_size + sample]
  float prediction[] = {
      1.0f, 2.0f,  // feature=0, samples=0..1
      3.0f, 4.0f,  // feature=1, samples=0..1
      5.0f, 6.0f   // feature=2, samples=0..1
  };
  float ground_truth[] = {0.0f, 2.5f, 3.0f, 3.5f, 5.5f, 5.0f};

  float loss = 0.0f;
  float gradient[6];
  memset(gradient, 0, sizeof(gradient));

  // Forward pass for MSE
  MSE_forward(prediction, ground_truth, batch_size, num_features, &loss);

  // Manual calculation for MSE:
  // sample 0 -> diffs: (1-0)^2 + (3-3)^2 + (5-5.5)^2 = 1 + 0 + 0.25 = 1.25
  // sample 1 -> diffs: (2-2.5)^2 + (4-3.5)^2 + (6-5)^2 = 0.25 + 0.5 + 1 = 1.75
  // total = 1.25 + 1.75 = 3.0
  // MSE = total / batch_size = 3.0 / 2 = 1.5
  float expected_loss = 1.5f;
  assert(fabsf(loss - expected_loss) < 1e-6 && "MSE_forward incorrect");

  // Backward pass for MSE
  MSE_backward(prediction, ground_truth, batch_size, num_features, gradient);

  // Expected gradient for MSE: 2 * (prediction - ground_truth)
  // sample 0 (feature=0) -> 2*(1-0) = 2
  // sample 1 (feature=0) -> 2*(2-2.5) = -1
  // sample 0 (feature=1) -> 2*(3-3) = 0
  // sample 1 (feature=1) -> 2*(4-3.5) = 1
  // sample 0 (feature=2) -> 2*(5-5.5) = -1
  // sample 1 (feature=2) -> 2*(6-5) = 2
  // Note the indexing (feature * batch_size + sample).
  float eps = 1e-6f;
  assert(fabsf(gradient[0] - 2.0f) < eps &&
         "MSE_backward gradient[0] incorrect");
  assert(fabsf(gradient[1] - (-1.0f)) < eps &&
         "MSE_backward gradient[1] incorrect");
  assert(fabsf(gradient[2] - 0.0f) < eps &&
         "MSE_backward gradient[2] incorrect");
  assert(fabsf(gradient[3] - 1.0f) < eps &&
         "MSE_backward gradient[3] incorrect");
  assert(fabsf(gradient[4] - (-1.0f)) < eps &&
         "MSE_backward gradient[4] incorrect");
  assert(fabsf(gradient[5] - 2.0f) < eps &&
         "MSE_backward gradient[5] incorrect");
}

/**
 * Tests for CrossEntropy_forward and CrossEntropy_backward
 */
static void test_CrossEntropy(void) {
  const int batch_size = 2;
  const int num_classes = 3;

  // Predictions (probabilities) arranged as prediction[class * batch_size +
  // sample]
  float linearized_prediction[] = {// class 0, samples 0..1
                                   0.1f, 0.3f,
                                   // class 1, samples 0..1
                                   0.7f, 0.3f,
                                   // class 2, samples 0..1
                                   0.2f, 0.4f};

  // One-hot ground truth
  // sample 0 -> class 1
  // sample 1 -> class 2
  float ground_truth[] = {// class 0
                          0.0f, 0.0f,
                          // class 1
                          1.0f, 0.0f,
                          // class 2
                          0.0f, 1.0f};

  float loss = 0.0f;
  CrossEntropy_forward(linearized_prediction, ground_truth, batch_size,
                       num_classes, &loss);

  // Manual approximation for CrossEntropy:
  // sample 0: -log(0.7 + 1e-9) ~ 0.356675
  // sample 1: -log(0.4 + 1e-9) ~ 0.916291
  // sum ~ 1.272966
  // average = 1.272966 / 2 ~ 0.636483
  float expected_loss = 0.636483f;
  assert(fabsf(loss - expected_loss) < 1e-5 &&
         "CrossEntropy_forward incorrect");

  float gradient[6];
  memset(gradient, 0, sizeof(gradient));
  CrossEntropy_backward(linearized_prediction, ground_truth, batch_size,
                        num_classes, gradient);

  // Expected gradient: prediction - ground_truth
  // class 0 (sample 0) => 0.1 - 0.0 = 0.1
  // class 0 (sample 1) => 0.3 - 0.0 = 0.3
  // class 1 (sample 0) => 0.7 - 1.0 = -0.3
  // class 1 (sample 1) => 0.3 - 0.0 = 0.3
  // class 2 (sample 0) => 0.2 - 0.0 = 0.2
  // class 2 (sample 1) => 0.4 - 1.0 = -0.6
  float expected_grad[] = {0.1f, 0.3f, -0.3f, 0.3f, 0.2f, -0.6f};

  float eps = 1e-6f;
  for (int i = 0; i < 6; i++) {
    assert(fabsf(gradient[i] - expected_grad[i]) < eps &&
           "CrossEntropy_backward gradient incorrect");
  }
}

/**
 * Main test entry point
 */
void test_loss(void) {
  test_MSE();
  test_CrossEntropy();
}
