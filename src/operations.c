#include "operations.h"

void matmul(float *A, const float *B, float *C, int M, int N, int K) {
  /*
      A: Pointer to the first matrix (MxK) in column-major format
      B: Pointer to the second matrix (KxN) in column-major format
      C: Pointer to the result matrix (MxN) in column-major format
      M: Number of rows in A and C
      N: Number of columns in B and C
      K: Number of columns in A and rows in B
  */

  // Convert uint to int for OpenBLAS
  int m = (int)M;
  int n = (int)N;
  int k = (int)K;

  int lda = (int)M;
  int ldb = (int)K;
  int ldc = (int)M;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda,
              B, ldb, beta, C, ldc);
}
