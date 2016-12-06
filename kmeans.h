#ifndef KMEANS_HEADER
#define KMEANS_HEADER

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <math.h>
#include <stdint.h>

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

/// @brief Performs K-means clustering on GPU / CUDA.
/// @param init centroids initialization method.
/// @param tolerance if the number of reassignments drop below this ratio, stop.
/// @param num_samples number of samples.
/// @param num_features number of features.
/// @param num_clusters number of clusters.
/// @param seed random generator seed passed to srand().
/// @param samples input array of sample points
/// @param centroids output array of centroids
/// @param assignments output array of cluster indices
/// @return cudaError_t
cudaError_t kmeans_cuda(InitMethod init, float tolerance, uint32_t num_samples,
                        uint32_t num_features, uint32_t num_clusters_size,
                        uint32_t seed, const float *samples, float *centroids,
                        uint32_t *memberships);

#endif
