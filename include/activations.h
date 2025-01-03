#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <omp.h>
#include <stddef.h>

void sigmoid_batch(const float* input, float* output, const int input_size,
                   const int batch_size);
void relu_batch(const float* input, float* output, const int input_size,
                const int batch_size);
void tanh_batch(const float* input, float* output, const int input_size,
                const int batch_size);
void leaky_relu_batch(const float* input, float* output, const int input_size,
                      const int batch_size, const float alpha);
void softmax_batch(const float* input, float* output, const int input_size,
                   const int batch_size);

#endif  // ACTIVATIONS_H
