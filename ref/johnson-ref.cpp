#include <assert.h>
#include <bits/stdc++.h>
#include <error.h>
#include <limits.h>
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

#define RC(i, j) (i * N + j)
#define INF (N * 100)

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
    for (int v = 0; v < N; v++) {
      int weight = get_G(u, v);
      if (weight != -1 && weight < INF && dist[u] + weight < dist[v]) {
        dist[v] = dist[u] + weight;
        pq.push(std::make_pair(-dist[v], v));
      }
    }
  }
  for (int v = 0; v < N; v++) {
    D[RC(s, v)] = dist[v];
  }
}

void apsp_start() {
  int* bellman_ford = (int*)malloc((N + 1) * sizeof(int));

  bellman_ford[N] = 0;
  for (int i = 0; i < N; i++) {
    bellman_ford[i] = INF;
  }

  for (int i = 0; i < N + 1; i++) {
    for (int u = 0; u < N + 1; u++) {
      for (int v = 0; v < N + 1; v++) {
        int weight;
        if (u == N || v == N) {
          weight = 0;
        } else {
          weight = get_G(u, v);
        }
        if (weight != -1 && weight < INF && bellman_ford[u] + weight < bellman_ford[v]) {
          bellman_ford[v] = bellman_ford[u] + weight;
        }
      }
    }
  }

  for (int u = 0; u < N; u++) {
    for (int v = 0; v < N; v++) {
      G[RC(u, v)] = get_G(u, v) + bellman_ford[u] - bellman_ford[v];
    }
  }

  for (int u = 0; u < N; u++) {
    dijkstra(u);
  }
}

int main(int argc, char** argv) {
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
  struct timespec before, after;
  clock_gettime(CLOCK_REALTIME, &before);
  apsp_start();
  clock_gettime(CLOCK_REALTIME, &after);
  double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
  FILE* out = NULL;
  if (argc > 2 && strcmp(argv[2], "-o") == 0) {
    out = fopen(argv[3], "w");
  }
  apsp_print_result(out);
  printf("Time: %.3f ms (%.3f s)\n", delta_ms, delta_ms / 1000.0);
  return 0;
}
