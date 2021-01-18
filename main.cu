/*
 *   Copyright (c) 2021
 *   All rights reserved.
 */
#include "cuda.h"

int main() {

  if (!initCUDA()) {
    printf("CUDA initialized fail.\n");
    return 0;
  } else {
    printf("CUDA initialized success.\n\n");
  }

  if (!calculateSumOfSquares()) {
    printf("calculate sum of squares fail.\n");
    return 0;
  } else {
    printf("calculate sum of squares success.\n\n");
  }

  return 0;
}