#include <assert.h>
#include <error.h>
#include <limits.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SYSEXPECT(expr) \
  do {                  \
    if (!(expr)) {      \
      perror(__func__); \
      exit(1);          \
    }                   \
  } while (0)
#define error_exit(fmt, ...)                                    \
  do {                                                          \
    fprintf(stderr, "%s error: " fmt, __func__, ##__VA_ARGS__); \
    exit(1);                                                    \
  } while (0);

int N = -1;     // number of vertices in the file.
int* G = NULL;  // adjacency matrix representation of the graph
int* D = NULL;  // matrix of distances
int NCORES = -1;

#define RC(i, j) (i * N + j)
#define INF (N * 100)
#define B 8
#define NB (N / B)

// set G[i,j] to value
inline static void set_G(int i, int j, int value) {
  assert(value >= 0 || value == -1);
  G[RC(i, j)] = (value >= 0) ? value : INF;
  return;
}

// returns value at G[i,j]
inline static int get_G(int i, int j) {
  return G[RC(i, j)];
}

void init_G(FILE* fp) {
  // Allocate memory and read the matrix
  G = (int*)malloc(N * N * sizeof(int));
  SYSEXPECT(G != NULL);
  int scan_ret;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int t;
      scan_ret = fscanf(fp, "%d", &t);
      if (scan_ret != 1)
        error_exit("Failed to read G(%d, %d)\n", i, j);
      set_G(i, j, t);
    }
  }
  fclose(fp);
}

// prints results
void apsp_print_result(FILE* out) {
  int sum1 = 0;
  int sum2 = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int d = D[RC(i, j)];
      if (d >= INF) {
        d = -1;
      }
      sum1 = (sum1 + d) % 255;
      sum2 = (sum2 + sum1) % 255;
      if (out != NULL) {
        fprintf(out, "%d ", d);
      }
    }
    if (out != NULL) {
      fprintf(out, "\n");
    }
  }
  printf("Checksum: %d\n", (sum2 << 8) | sum1);
  if (out != NULL) {
    fclose(out);
  }
  return;
}

inline void floyd_block(int bi, int bj, int bk) {
  for (int k = bk * B; k < bk * B + B; k++) {
    for (int i = bi * B; i < bi * B + B; i++) {
      for (int j = bj * B; j < bj * B + B; j++) {
        int d = D[RC(i, k)] + D[RC(k, j)];
        if (d < D[RC(i, j)]) {
          D[RC(i, j)] = d;
        }
      }
    }
  }
}

void apsp_start() {
#pragma omp parallel for num_threads(NCORES)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      D[RC(i, j)] = get_G(i, j);
    }
  }

  for (int bk = 0; bk < NB; bk++) {
    floyd_block(bk, bk, bk);
#pragma omp parallel for num_threads(NCORES)
    for (int bj = 0; bj < NB; bj++) {
      if (bj == bk)
        continue;
      floyd_block(bk, bj, bk);
    }

#pragma omp parallel for num_threads(NCORES)
    for (int bi = 0; bi < NB; bi++) {
      if (bi == bk)
        continue;
      floyd_block(bi, bk, bk);
      for (int bj = 0; bj < NB; bj++) {
        if (bj == bk)
          continue;
        floyd_block(bi, bj, bk);
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 4 || strcmp(argv[1], "-p") != 0)
    error_exit(
        "Expecting two arguments: -p [processor count] and [file name]\n");
  NCORES = atoi(argv[2]);
  if (NCORES < 1)
    error_exit("Illegal core count: %d\n", NCORES);
  char* filename = argv[3];
  FILE* fp = fopen(filename, "r");
  if (fp == NULL)
    error_exit("Failed to open input file \"%s\"\n", filename);
  int scan_ret;
  scan_ret = fscanf(fp, "%d", &N);
  if (scan_ret != 1)
    error_exit("Failed to read vertex count\n");
  if (N < 2) {
    error_exit("Illegal vertex count: %d\n", N);
  }
  init_G(fp);
  D = (int*)malloc(N * N * sizeof(int));
  SYSEXPECT(D != NULL);
  struct timespec before, after;
  clock_gettime(CLOCK_REALTIME, &before);
  apsp_start();
  clock_gettime(CLOCK_REALTIME, &after);
  double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 +
                    (after.tv_nsec - before.tv_nsec) / 1000000.0;
  FILE* out = NULL;
  if (argc > 4 && strcmp(argv[4], "-o") == 0) {
    out = fopen(argv[5], "w");
  }
  apsp_print_result(out);
  printf("Time: %.3f ms (%.3f s)\n", delta_ms, delta_ms / 1000.0);
  return 0;
}
