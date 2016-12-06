#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include "kmeans.h"

//Define these constant variables that are going to be that way for the entire
//experiment
__constant__ uint32_t num_features_dim;
__constant__ uint32_t num_samples;
__constant__ uint32_t num_clusters;
__constant__ uint32_t shmem_size;


__device__ float distance( float *sample1, float * sample2, uint32_t
        incr1, uint32_t incr2);


cudaError_t kmeans_cuda( InitMethod init, float tolerance, uint32_t num_samples,
        uint32_t num_features, uint32_t num_clusters_size, uint32_t seed, const
        float *samples, float *centroids, uint32_t *memberships)
{
    return cudaSuccess;
}

// Currently handled as an array, contingent upon caller to coalesce access.
// sample1 - memory location from which to read the data for point 1. Typically
// in our case going to be the dataset we are looking at.
// sample2 - memory location from which to read the data for point 2. Typicaly
// in our case going to be the centroids we are looking at.
// incr1 - memory jumps to access next feature of point 1. Typically in our case
// is going to be the dataset size.
// incr2 - memory jumps to access next feature of point 2. Typically in our case
// is going to be the number of centroids K.
// num_features_dim - assumed to be in constant memory indicates feature dimension
// space
__device__ float distance( float *sample1, float *sample2, uint32_t incr1,
        uint32_t incr2)
{
    int ret_distance = 0;
    for(int i=0;i<num_features_dim;i++)
        ret_distance +=
            (sample1[i*incr1]-sample2[i*incr2])*(sample1[i*incr1]-sample2[i*incr2]);
    return ret_distance;
}

__global__ void nearest_cluster_assign( float *samples, float *centroids,
        uint32_t *membership)
{
    uint32_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float min_dist = FLT_MAX;
    uint32_t nearest_cluster = num_clusters-1;
    if(sample_idx >= num_samples)
        return;
    // Goto concerned start pointer in samples array
    samples += sample_idx;
    //Create a shared mem buffer for centroid. If all centroids fit into the
    //region, well and good. If not we load it on and off in batches and take
    //distances. Theoretically should benefit from it but yet to see if it
    //actually deteriorates emperically.
    extern __shared__ float shared_centroids[];
    //TODO: Check back to see if this calculation is right or sizeof(float)
    //incorporated elsewhere
    const uint32_t max_shared_centroids =
        shmem_size/(num_features_dim*sizeof(float));
    //TODO: define min if needed
    const uint32_t thread_num_shared_process =
        ceilf(max_shared_centroids/min(blockDim.x, num_samples - blockIdx.x *
                blockDim.x));
    //Load a batch of centroids to shared and compute pairwise distance between
    //the current point and all centroids
    for(uint32_t centroids_batch=0; centroids_batch<num_clusters;
            centroids_batch += max_shared_centroids)
    {
        for(uint32_t i=0; i<thread_num_shared_process; i++)
        {
            uint32_t local_offset = i * thread_num_shared_process + threadIdx.x;
            uint32_t global_offset = local_offset + centroids_batch;
            //Confused in offsets, put a conditional here to be safe
            if(global_offset<num_clusters && local_offset<max_shared_centroids)
            {
                for(uint32_t feature_idx=0; feature_idx<num_features_dim;
                        feature_idx++)
                {
                    shared_centroids[feature_idx*num_clusters + local_offset] =
                        centroids[feature_idx*num_clusters + global_offset];
                }
            }
        }
        __syncthreads();
        for(uint32_t cluster = centroids_batch; cluster < centroids_batch +
                max_shared_centroids && cluster < num_clusters; cluster++)
        {
            float dist = distance(samples, shared_centroids+cluster,
                    num_samples, max_shared_centroids);
            if(dist<min_dist)
            {
                min_dist = dist;
                nearest_cluster = cluster;
            }
        }
    }
    membership[sample_idx] = nearest_cluster;
}
