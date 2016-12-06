#include <stdlib.h>
#include <stdio.h>
#include <cmath>

#include "kmeans.h"

cudaError_t kmeans_cuda( InitMethod init, float tolerance, uint32_t n_samples,
        uint32_t n_features, uint32_t n_clusters_size, uint32_t seed, const
        float *samples, float *centroids, uint32_t *memberships)
{
    return cudaSuccess;
}
