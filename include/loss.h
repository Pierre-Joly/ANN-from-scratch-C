#ifndef LOSS_H
#define LOSS_H

#include <math.h>
#include <omp.h>

void MSE_forward(const float* prediction, const float* ground_truth,
                 int batch_size, int num_features, float* loss);
void MSE_backward(const float* prediction, const float* ground_truth,
                  int batch_size, int num_features, float* gradient);
void CrossEntropy_forward(const float* prediction, const float* ground_truth,
                          int batch_size, int num_classes, float* loss);
void CrossEntropy_backward(const float* prediction, const float* ground_truth,
                           int batch_size, int num_classes, float* gradient);

#endif  // LOSS_H
