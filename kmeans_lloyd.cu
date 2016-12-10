#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include "kmeans.h"

#define BLOCK_SZ_CNT_ASS 256
#define BLOCK_SZ_CNT_ADJ 256

//Define some counters on device for access to changes
__device__ uint32_t mem_change_ctr;
//Define these constant variables that are going to be that way for the entire
//experiment
__constant__ uint32_t num_features;
__constant__ uint32_t num_samples;
__constant__ uint32_t num_clusters;
__constant__ uint32_t shmem_size;

// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
__device__ __forceinline__ uint32_t atomicAggInc(uint32_t *ctr) {
  int mask = __ballot(1);
  int leader = __ffs(mask) - 1;
  uint32_t res;
  if ((threadIdx.x % 32) == leader) {
    res = atomicAdd(ctr, __popc(mask));
  }
  res = __shfl(res, leader);
  return res + __popc(mask & ((1 << (threadIdx.x % 32)) - 1));
}


__device__ float distance( float *sample1, float * sample2, uint32_t
        incr1, uint32_t incr2);


// Currently handled as an array, contingent upon caller to coalesce access.
// sample1 - memory location from which to read the data for point 1. Typically
// in our case going to be the dataset we are looking at.
// sample2 - memory location from which to read the data for point 2. Typicaly
// in our case going to be the centroids we are looking at.
// incr1 - memory jumps to access next feature of point 1. Typically in our case
// is going to be the dataset size.
// incr2 - memory jumps to access next feature of point 2. Typically in our case
// is going to be the number of centroids K.
// num_features - assumed to be in constant memory indicates feature dimension
// space
__device__ float distance( float *sample1, float *sample2, uint32_t incr1,
        uint32_t incr2)
{
    int ret_distance = 0;
    for(int i=0;i<num_features;i++)
        ret_distance +=
            (sample1[i*incr1]-sample2[i*incr2])*(sample1[i*incr1]-sample2[i*incr2]);
    return ret_distance;
}

__global__ void nearest_cluster_assign( float *samples, float *centroids,
        uint32_t *membership, uint32_t *membership_old)
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
        shmem_size/(num_features*sizeof(float));
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
                for(uint32_t feature_idx=0; feature_idx<num_features;
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
    uint32_t mem_old = membership[sample_idx];
    membership_old[sample_idx] = mem_old;
    if(mem_old != nearest_cluster)
    {
        membership[sample_idx] = nearest_cluster;
        atomicAggInc(&mem_change_ctr);
    }
}

__global__ void adjust_centroids( float *samples, float *centroids, uint32_t
        *membership, uint32_t *membership_old, uint32_t *cluster_counts)
{
    uint32_t centroid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(centroid_idx >= num_clusters)
        return;
    centroids += centroid_idx;
    uint32_t cluster_count = cluster_counts[centroid_idx];
    //multiply each centroid by it's count to make it ready for adjustments -
    //neccessary evil of globalmemory writes
    for(uint32_t i = 0; i < num_features; i++)
    {
        centroids[i*num_clusters] *= cluster_count;
    }
    extern __shared__ uint32_t shared_memberships[];
    //Since each membership is of type uint32_t and we have old and new
    //memberships, we can load the memberships to fill half the shared memory
    uint32_t sample_step = shmem_size/(2*sizeof(uint32_t));
    uint32_t samples_per_thread = ceilf(sample_step/min(blockDim.x, num_clusters
                - blockDim.x * blockDim.x));
    for(uint32_t sample_start = 0; sample_start < num_samples; sample_start +=
            sample_step)
    {
        for(uint32_t i=0; i<samples_per_thread; i++)
        {
            uint32_t local_offset = i * samples_per_thread + threadIdx.x;
            uint32_t global_offset = local_offset + sample_start;
            if(global_offset < num_samples && local_offset < sample_step)
            {
                shared_memberships[2*local_offset] = membership[global_offset];
                shared_memberships[2*local_offset+1] = membership_old[global_offset];
            }
        }
        __syncthreads();
        //Now each thread is going to scan all the shared samples
        for(uint32_t i=0; i < sample_step && sample_start + i < num_samples;
                i++)
        {
            uint32_t local_membership = shared_memberships[2*i];
            uint32_t local_membership_prev = shared_memberships[2*i+1];
            int sign = 0;
            if(local_membership_prev == centroid_idx && local_membership !=
                    centroid_idx)
            {
                sign = -1;
                cluster_count--;
            }
            else if(local_membership_prev != centroid_idx && local_membership ==
                    centroid_idx)
            {
                sign = 1;
                cluster_count++;
            }
            if(sign)
            {
                uint32_t sample_offset = sample_start + i;
                for(uint32_t feature = 0; feature < num_features; feature++)
                {
                    centroids[feature * num_clusters] += sign *
                        samples[sample_offset + feature * num_samples];
                }
            }
        }
    }
    // Average the centroid
    for(uint32_t i = 0; i < num_features; i++)
        centroids[i*num_clusters] /= cluster_count;
    //Write back local count to memory
    cluster_counts[centroid_idx] = cluster_count;
}

//------------------------Host Functions--------------------------------

uint32_t initTasks(uint32_t n_samples, uint32_t n_clusters, uint32_t
        n_features, int dev_num=0)
{
    gpuErrchk(cudaMemcpyToSymbol(num_samples, &n_samples, sizeof(n_samples)));
    gpuErrchk(cudaMemcpyToSymbol(num_clusters, &n_clusters, sizeof(n_clusters)));
    gpuErrchk(cudaMemcpyToSymbol(num_features, &n_features, sizeof(n_features)));
    cudaDeviceProp props;
    gpuErrchk(cudaSetDevice(dev_num));
    gpuErrchk(cudaGetDeviceProperties(&props, dev_num));
    uint32_t smem_size = props.sharedMemPerBlock;
    printf("gpu %d has %d bytes of shared memory\n", dev_num, smem_size); 
    gpuErrchk(cudaMemcpyToSymbol(shmem_size, &smem_size, sizeof(smem_size)));
    uint32_t zero = 0;
    gpuErrchk(cudaMemcpyToSymbol(mem_change_ctr, &zero, sizeof(zero)));
    return smem_size;
}

int check_change_ratio(float tolerance, uint32_t n_samples)
{
    uint32_t num_changes = 0;
    gpuErrchk(cudaMemcpyFromSymbol(&num_changes, mem_change_ctr,
                sizeof(num_changes)));
    printf("numchanges = %d\n",num_changes);
    if(num_changes <= tolerance * n_samples)
        return -1;
    uint32_t zero = 0;
    gpuErrchk(cudaMemcpyToSymbol(mem_change_ctr, &zero, sizeof(zero)));
    return 0;
}

cudaError_t kmeans_cuda( InitMethod init, float tolerance, uint32_t n_samples,
        uint32_t n_features, uint32_t n_clusters, uint32_t seed, float
        *samples, float *centroids, uint32_t *memberships, int *iterations =
        NULL)
{
    uint32_t smem_size = initTasks(n_samples, n_clusters, n_features);
    dim3 sample_block(BLOCK_SZ_CNT_ASS);
    dim3 centroid_block(BLOCK_SZ_CNT_ADJ);
    dim3 sample_grid(ceil(1.0 * n_samples/sample_block.x));
    dim3 centroid_grid(ceil(1.0 * n_clusters/centroid_block.x));
    uint32_t *memberships_old, *cluster_counts;
    gpuErrchk(cudaMalloc((void **) &memberships_old,
                n_samples*sizeof(uint32_t)));
    gpuErrchk(cudaMalloc((void **) &cluster_counts,
                n_clusters*sizeof(uint32_t)));
    gpuErrchk(cudaMemset(cluster_counts, 0, n_clusters*sizeof(uint32_t)));
    //arbitrary - set maxiter to 500
    for(int i = 0; i < 500; i++)
    {
        printf("grid size is %dx%d, block size is %dx%d and shared mem needed is %d\n"
                , sample_grid.x, sample_grid.y, sample_block.x,
                sample_block.y, smem_size);
        nearest_cluster_assign<<<sample_grid,sample_block,smem_size>>>( samples,
                centroids, memberships, memberships_old);
        gpuErrchk( cudaPeekAtLastError() );
        int change_ratio_good = check_change_ratio(tolerance, n_samples);
        printf("change ratio is %d\n",change_ratio_good);
        if(change_ratio_good<0)
        {
            if(iterations)
                *iterations = i;
            return cudaSuccess;
        }
        adjust_centroids<<<centroid_grid,centroid_block,smem_size>>>( samples,
                centroids, memberships, memberships_old, cluster_counts);
        gpuErrchk( cudaPeekAtLastError() );
    }
    return cudaSuccess;
}
