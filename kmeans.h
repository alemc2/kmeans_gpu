#ifndef KMEANS_HEADER
#define KMEANS_HEADER

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N) ((i) * (N)) + (j)

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) exit(code);
    }
}

enum InitMethod { InitMethodRandom = 0, InitMethodPlusPlus, InitMethodImport };

__host__ __device__ int copy_vectors(float *dst_vec, float *src_vec,
                                     const uint32_t n_dim,
                                     const uint32_t dst_skip,
                                     const uint32_t src_skip);

float **file_read(int, char *, uint32_t *, uint32_t *);
int file_write(char *, uint32_t, uint32_t, uint32_t, float **, uint32_t *);

float *transpose(float *src, uint32_t n_samples, uint32_t n_features);
void print1d(uint32_t *src, uint32_t dim);
void print2d(float *src, uint32_t dim1, uint32_t dim2);

/// @brief Performs K-means clustering on GPU / CUDA.
/// @param init centroids initialization method.
/// @param tolerance if the number of reassignments drop below this ratio, stop.
/// @param num_samples number of samples.
/// @param num_features number of features.
/// @param num_clusters number of clusters.
/// @param seed random generator seed passed to srand().
/// @param samples input array of sample points
/// @param centroids output array of centroids
/// @param memberships array of cluster indices
/// @return cudaError_t
//
// Assume samples,centroids,memberships to be already initialized arrays in
// device
cudaError_t kmeans_cuda(InitMethod init, float tolerance, uint32_t n_samples,
                        uint32_t n_features, uint32_t n_clusters, uint32_t seed,
                        float *samples, float *centroids, uint32_t *memberships,
                        int *iterations);

int init_centroids(InitMethod method, uint32_t n_samples, uint32_t n_features,
                   uint32_t n_clusters, uint32_t seed, float *h_samples,
                   float *h_centroids);

extern int _debug;
#endif
