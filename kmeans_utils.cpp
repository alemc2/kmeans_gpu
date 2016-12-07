
#include "kmeans.h"

void shuffle(uint32_t *array, uint32_t n) {
    if (n > 1) {
        uint32_t i;
        for (i = 0; i < n - 1; i++) {
            uint32_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            uint32_t t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

__host__ __device__ int copy_vectors(float *dst_vec, float *src_vec,
                                     const uint32_t n_dim,
                                     const uint32_t dst_skip,
                                     const uint32_t src_skip) {
    for (uint32_t i = 0; i < n_dim; i++)
        dst_vec[i * dst_skip] = src_vec[i * src_skip];
    return 0;
}

int init_centroids(InitMethod method, uint32_t n_samples, uint32_t n_features,
                   uint32_t n_clusters, uint32_t seed, float *h_samples,
                   float *h_centroids) {
    srand(seed);
    uint32_t selection[n_samples];
    switch (method) {
        case InitMethodImport:
            return 0;
            break;
        case InitMethodRandom:
            for (uint32_t i = 0; i < n_samples; i++) selection[i] = i;
            shuffle(selection, n_samples);
            for (uint32_t i = 0; i < n_clusters; i++)
                copy_vectors(h_centroids + i, h_samples + selection[i],
                             n_features, n_clusters, n_samples);
            break;
        case InitMethodPlusPlus:
            break;
    }
    return 0;
}
