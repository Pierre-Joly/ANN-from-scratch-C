#ifndef ERRORS_H
#define ERRORS_H

// General error codes
#define SUCCESS 0                     // Operation succeeded
#define ERR_OUT_OF_BOUNDS -1          // Index is out of bounds
#define ERR_LAYER_NOT_INITIALIZED -2  // Preceding layer not initialized
#define ERR_LAYER_ALREADY_SET -3      // Layer already initialized
#define ERR_BATCH_SIZE_MISMATCH -4    // Batch size mismatch
#define ERR_DIMENSION_MISMATCH -5     // Dimensions are not compatible

#endif  // ERRORS_H
