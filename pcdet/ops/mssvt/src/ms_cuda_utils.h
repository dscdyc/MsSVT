#ifndef MS_CUDA_UTILS_H
#define MS_CUDA_UTILS_H

#include <cmath>

#define THREADS_PER_BLOCK 256
// #define THREADS_PER_BLOCK 1024
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
#define EMPTY_KEY -1
#define BLK_SIGNAL -2

#endif