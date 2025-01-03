#include "loss.h"

void MSE_forward(const float* prediction, const float* ground_truth,
                 int batch_size, int num_features, float* loss) {
  float local_loss = 0.0f;
#pragma omp parallel for collapse(2) reduction(+ : local_loss)
  for (int sample = 0; sample < batch_size; sample++) {
    for (int feature = 0; feature < num_features; feature++) {
      int idx = feature * batch_size + sample;
      float diff = (prediction[idx] - ground_truth[idx]);
      local_loss += diff * diff;
    }
  }
  *loss = local_loss / (float)(batch_size);
}

void MSE_backward(const float* prediction, const float* ground_truth,
                  int batch_size, int num_features, float* gradient) {
#pragma omp parallel for collapse(2)
  for (int sample = 0; sample < batch_size; sample++) {
    for (int feature = 0; feature < num_features; feature++) {
      int idx = feature * batch_size + sample;
      gradient[idx] = 2 * (prediction[idx] - ground_truth[idx]);
    }
  }
}

void CrossEntropy_forward(const float* prediction, const float* ground_truth,
                          int batch_size, int num_classes, float* loss) {
  float local_loss = 0.0f;
#pragma omp parallel for collapse(2) reduction(- : local_loss)
  for (int sample = 0; sample < batch_size; sample++) {
    for (int classe = 0; classe < num_classes; classe++) {
      int idx = classe * batch_size + sample;
      local_loss -= ground_truth[idx] * logf(prediction[idx] + 1e-9f);
    }
  }
  *loss = local_loss / (float)(batch_size);
}

void CrossEntropy_backward(const float* prediction, const float* ground_truth,
                           int batch_size, int num_classes, float* gradient) {
#pragma omp parallel for collapse(2)
  for (int sample = 0; sample < batch_size; sample++) {
    for (int classe = 0; classe < num_classes; classe++) {
      int idx = classe * batch_size + sample;
      gradient[idx] = prediction[idx] - ground_truth[idx];
    }
  }
}
