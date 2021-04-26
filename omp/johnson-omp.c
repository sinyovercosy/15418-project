#include <assert.h>
#include <error.h>
#include <limits.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <bits/stdc++.h>
#include <boost/heap/fibonacci_heap.hpp>

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
#define INF (N * N * 100)

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
void apsp_print_result() {
  int sum1 = 0;
  int sum2 = 0;
  printf("========== Solution ==========\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int d = D[RC(i, j)];
      if (d >= INF) {
        d = -1;
      }
      sum1 = (sum1 + d) % 255;
      sum2 = (sum2 + sum1) % 255;
      // printf("%d ", d);
    }
    // putchar('\n');
  }
  putchar('\n');
  printf("Checksum: %d", (sum2 << 8) | sum1);
  putchar('\n');
  return;
}

int* modified_G = NULL;

void dijkstra(int s) {
  int* dist = (int*)malloc(sizeof(int) * N);
  for (int i = 0; i < N; i++) {
    dist[i] = INF;
  }
  dist[s] = 0;

  std::priority_queue<std::pair<int, int> > pq;

  pq.push(std::make_pair(0, s));

  while(!pq.empty()) {
    int u = pq.top().second;
    pq.pop();
    for (int v = 0; v < N && modified_G[RC(u,v)] != -1; v++) {
      int real_v = modified_G[RC(u,v)];
      int weight = get_G(u, real_v);
      if (dist[u] + weight < dist[real_v]) {
        dist[real_v] = dist[u] + weight;
        pq.push(std::make_pair(-dist[real_v], real_v));
      }
    }
  }
  for (int v = 0; v < N; v++) {
    D[RC(s, v)] = dist[v];
  }
}


void apsp_start() {

  modified_G = (int*) malloc(N*N*sizeof(int));

// THIS ONE SOMEHOW MAKES IT SLOWER
// #pragma omp parallel for num_threads(NCORES) 
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

  int* bellman_ford = (int*) malloc((N + 1) * sizeof(int));

  bellman_ford[N] = 0;
  for (int i = 0; i < N; i++) {
    bellman_ford[i] = INF;
  }

  for (int i = 0; i < N + 1; i++) {
#pragma omp parallel for num_threads(NCORES)
    for (int u = 0; u < N + 1; u++) {
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
        if (bellman_ford[u] < bellman_ford[N]) {
          bellman_ford[N] = bellman_ford[u];
        }
      }
    }
  }
  

#pragma omp parallel for num_threads(NCORES)
  for (int u = 0; u < N; u++) {
    for (int v = 0; v < N && modified_G[RC(u, v)] != -1; v++) {
      int real_v = modified_G[RC(u, v)];
      G[RC(u, real_v)] = get_G(u, real_v) + bellman_ford[u] - bellman_ford[real_v];
    }
  }

#pragma omp parallel for num_threads(NCORES)
  for (int u = 0; u < N; u++) {
    dijkstra(u);
  }
}


int main(int argc, char** argv) {
  if (argc < 4 || strcmp(argv[1], "-p") != 0)
    error_exit("Expecting two arguments: -p [processor count] and [file name]\n");
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
  printf("helon\n");
  D = (int*)malloc(N * N * sizeof(int));
  SYSEXPECT(D != NULL);
  struct timespec before, after;
  clock_gettime(CLOCK_REALTIME, &before);
  apsp_start();
  clock_gettime(CLOCK_REALTIME, &after);
  double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
  putchar('\n');
  printf("============ Time ============\n");
  printf("Time: %.3f ms (%.3f s)\n", delta_ms, delta_ms / 1000.0);
  apsp_print_result();
  return 0;
}
