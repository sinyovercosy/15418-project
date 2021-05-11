#include <assert.h>
#include <error.h>
#include <limits.h>
#include <mpi.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

#define RC(i, j) (i * N + j)
#define INF (N * 100)
#define ROOT 0
#define TAG 0

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

void apsp_start(int procID, int nproc) {
  int span = (N + nproc - 1) / nproc;
  int startRow = procID * span;
  int endRow = startRow + span;
  if (endRow > N) {
    endRow = N;
  }

  int bufsize = span * nproc;
  int* kCol = (int*)malloc(bufsize * sizeof(int));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      D[RC(i, j)] = get_G(i, j);
    }
    kCol[i] = D[RC(i, 0)];
  }

  double commTime = 0;
  for (int k = 0; k < N; k++) {
    // double startTime = MPI_Wtime();
    MPI_Bcast(D + k * N, N, MPI_INT, k / span, MPI_COMM_WORLD);
    MPI_Allgather(kCol + startRow, span, MPI_INT, kCol, span, MPI_INT,
                  MPI_COMM_WORLD);
    // double endTime = MPI_Wtime();
    // commTime += endTime - startTime;

    for (int i = startRow; i < endRow; i++) {
      for (int j = 0; j < N; j++) {
        int d = kCol[i] + D[RC(k, j)];
        if (d < D[RC(i, j)]) {
          D[RC(i, j)] = d;
        }
      }
      kCol[i] = D[RC(i, k + 1)];
    }
  }

  // double startTime = MPI_Wtime();
  MPI_Gather(D + startRow * N, span * N, MPI_INT, D, span * N, MPI_INT, ROOT,
             MPI_COMM_WORLD);
  // double endTime = MPI_Wtime();
  // commTime += endTime - startTime;
  // printf("Proc %d comm time: %.3f ms (%.3f s)\n", procID, commTime * 1000.0,
  //        commTime);
}

int main(int argc, char** argv) {
  int procID;
  int nproc;

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &procID);

  // Get total number of processes specificed at start of run
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (argc < 2)
    error_exit("Expecting argument [file name]\n");
  char* filename = argv[1];
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

  MPI_Barrier(MPI_COMM_WORLD);

  // Run computation
  double startTime = MPI_Wtime();
  if (nproc <= 2)
    apsp_start(procID, nproc);
  else
    apsp_start(procID, nproc);
  double endTime = MPI_Wtime();

  // Compute running time
  MPI_Finalize();
  if (procID == ROOT) {
    FILE* out = NULL;
    if (argc > 2 && strcmp(argv[2], "-o") == 0) {
      out = fopen(argv[3], "w");
    }
    apsp_print_result(out);
    double delta_s = endTime - startTime;
    printf("Time: %.3f ms (%.3f s)\n", delta_s * 1000.0, delta_s);
  }
  return 0;
}
