
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

// src is assumed to be a n_samplesxn_features matrix
float *transpose(float *src, uint32_t n_samples, uint32_t n_features,
                 int pinned_result) {
    float *dst;
    if (pinned_result) {
        gpuErrchk(cudaMallocHost((void **)&dst,
                                 n_samples * n_features * sizeof(float)));
    } else {
        dst = (float *)malloc(n_samples * n_features * sizeof(float));
    }
    uint32_t i, j;
    for (i = 0; i < n_features; i++) {
        for (j = 0; j < n_samples; j++) {
            dst[index(i, j, n_samples)] = src[index(j, i, n_features)];
        }
    }
    return dst;
}

void print1d(uint32_t *src, uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) printf("%u ", src[i]);
    printf("\n");
}

void print2d(float *src, uint32_t dim1, uint32_t dim2) {
    for (uint32_t i = 0; i < dim1; i++) {
        for (uint32_t j = 0; j < dim2; j++) {
            printf("%f ", src[index(i, j, dim2)]);
        }
        printf("\n");
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
