#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <cblas.h>
#include <stdio.h>

void matmul(float *A, const float *B, float *C, int M, int N, int K);

#endif  // OPERATIONS_H
