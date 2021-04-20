#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define error_exit(fmt, ...)                                    \
  do {                                                          \
    fprintf(stderr, "%s error: " fmt, __func__, ##__VA_ARGS__); \
    exit(1);                                                    \
  } while (0);
#define RANDOMRANGE 100
#define BUCKETSIZE ((RAND_MAX + 1u) / RANDOMRANGE)

int N;
float sparseness;

int main(int argc, char** argv) {
  if (argc < 3)
    error_exit("Expecting two arguments: [# vertices] and [sparseness]\n");
  N = atoi(argv[1]);
  if (N < 2)
    error_exit("Too few vertices\n");
  sparseness = atof(argv[2]);
  srand(time(NULL));

  printf("%d\n", N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j) {
        printf("  0 ");
      } else if (((float)rand() / RAND_MAX) < sparseness) {
        printf(" -1 ");
      } else {
        printf("%3d ", 1 + rand() / BUCKETSIZE);
      }
    }
    printf("\n");
  }
  return 0;
}
