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
#include <bits/stdc++.h>

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
  G[RC(i, j)] = (value != -1) ? value : INF;
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


int* modified_G = NULL;
int* bellman_ford = NULL;

void dijkstra(int s) {
  int* dist = (int*)malloc(sizeof(int) * N);
  for (int i = 0; i < N; i++) {
    dist[i] = INF;
  }
  dist[s] = 0;

  std::priority_queue<std::pair<int, int> > pq;

  pq.push(std::make_pair(0, s));

  while (!pq.empty()) {
    int u = pq.top().second;
    pq.pop();
    for (int v = 0; v < N && modified_G[RC(u, v)] != -1; v++) {
      int real_v = modified_G[RC(u, v)];
      int weight = get_G(u, real_v);
      if (dist[u] + weight < dist[real_v]) {
        dist[real_v] = dist[u] + weight;
        pq.push(std::make_pair(-dist[real_v], real_v));
      }
    }
  }
  for (int v = 0; v < N; v++) {
    D[RC(s, v)] = dist[v] + bellman_ford[v] - bellman_ford[s];
  }
}

void apsp_start(int procID, int nproc) {
  // if (procID == 0) {

  modified_G = (int*)malloc(N * N * sizeof(int));

  for (int i = 0; i < N; i++) {
    int counter = 0;
    for (int j = 0; j < N; j++) {
      if (get_G(i, j) < INF) {
        modified_G[RC(i, counter)] = j;
        counter++;
      }
    }
    if (counter < N) {
      modified_G[RC(i, counter)] = -1;
    }
  }
    
  int span_bf = (N + nproc) / nproc;
  int start_bf = procID * span_bf;
  int end_bf = start_bf + span_bf;
  if (end_bf > N + 1) {
    end_bf = N + 1;
  }

  bellman_ford = (int*)malloc((N + 1) * sizeof(int));
  int* bellman_ford_from_others = (int*)malloc(span_bf * (N + 1) * nproc * sizeof(int));

  bellman_ford[N] = 0;
  for (int i = 0; i < N; i++) {
    bellman_ford[i] = 0;
  }

  for (int i = 0; i < N + 1; i++) {
// #pragma omp parallel for num_threads(NCORES)
    for (int u = start_bf; u < end_bf; u++) {
      if (u == N) {
        for (int v = 0; v < N; v++) {
          if (bellman_ford[u] < bellman_ford[v]) {
            bellman_ford[v] = bellman_ford[u];
          }
        }
      } else {
        for (int v = 0; v < N && modified_G[RC(u, v)] != -1; v++) {
          int real_v = modified_G[RC(u, v)];
          int weight = get_G(u, real_v);
          if (bellman_ford[u] + weight < bellman_ford[real_v]) {
            bellman_ford[real_v] = bellman_ford[u] + weight;
          }
        }
      }
    }
  
    MPI_Allgather(bellman_ford, N + 1, MPI_INT, bellman_ford_from_others, N + 1, MPI_INT,
                  MPI_COMM_WORLD);
    for (int j = 0; j < nproc; j++) {
      for (int u = 0; u < N; u++) {
        if (bellman_ford_from_others[RC(j, u)] < bellman_ford[u]) {
          bellman_ford[u] = bellman_ford_from_others[RC(j, u)];
        }
      }
    }
  }


  for (int u = 0; u < N; u++) {
    for (int v = 0; v < N && modified_G[RC(u, v)] != -1; v++) {
      int real_v = modified_G[RC(u, v)];
      G[RC(u, real_v)] = get_G(u, real_v) + bellman_ford[u] - bellman_ford[real_v];
    }
  }

  int span_d = (N + nproc - 1) / nproc;
  int start_d = procID * span_d;
  int end_d = start_d + span_d;
  if (end_d > N) {
    end_d = N;
  }

  for (int u = start_d; u < end_d; u++) {
    dijkstra(u);
  }
  MPI_Gather(D + start_d * N, span_d * N, MPI_INT, D, span_d * N, MPI_INT, 0,
                  MPI_COMM_WORLD);
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
  int span = (N + nproc - 1) / nproc;
  int padded_N = span * nproc;
  D = (int*)malloc(padded_N * N * sizeof(int));
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
