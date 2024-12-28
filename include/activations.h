#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <omp.h>

void sigmoid_batch(const float *restrict input, float *restrict output,
                   const int batch_size, const int num_batches);
void relu_batch(const float *restrict input, float *restrict output,
                const int batch_size, const int num_batches);
void tanh_batch(const float *restrict input, float *restrict output,
                const int batch_size, const int num_batches);
void leaky_relu_batch(const float *restrict input, float *restrict output,
                      const int batch_size, const int num_batches,
                      const float alpha);
void softmax_batch(const float *restrict input, float *restrict output,
                   const int batch_size, const int num_batches);

#endif  // ACTIVATIONS_H
