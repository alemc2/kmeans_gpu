#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <cmath>

#include "kmeans.h"

using namespace std;

int _debug;

void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -b             : input file is in binary format (default no)- if \
        cluster file provided it should be in same format\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -c clusters    : file containg clusters to initialize to\n"
        "       -o             : output timing results (default no)\n"
        "       -d             : enable debug mode\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}

int main(int argc, char * argv[])
{
    int opt;
    extern char *optarg;
    extern int optind;
    int isBinaryFile, is_output_timing;
    Init_Method cluster_method;
    char *cluster_file;
    uint32_t numClusters, numFeatures, numSamples;
    uint32_t numClusters_read,numCluster_Features_read;
    uint32_t *membership;
    char *filename;
    float **samples;
    float *samples_T;
    float *clusters;
    float **clusters_2d;
    float *clusters_T;
    float *d_samples, *d_clusters;
    uint32_t *d_memberships;
    int numIterations;
    float threshold;
    uint32_t seed;

    // Time realted variables
    double io_time,cuda_time,compute_time;
    clock_t io_start, io_end, cuda_start, cuda_end, compute_start, compute_end;
    
    /* some default values */
    _debug           = 0;
    threshold        = 0.001;
    numClusters      = 0;
    isBinaryFile     = 0;
    is_output_timing = 0;
    cluster_method   = InitMethodRandom;
    cluster_file     = NULL;
    filename         = NULL;

    while ( (opt=getopt(argc,argv,"i:n:t:c:d:abo"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'c': cluster_method = InitMethodImport;
                      cluster_file = optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            case 'o': is_output_timing = 1;
                      break;
            case 'd': _debug = atoi(optarg);
                      break;
            case '?': usage(argv[0], threshold);
                      break;
            default: usage(argv[0], threshold);
                      break;
        }
    }

    if (filename == 0 || numClusters <= 1) usage(argv[0], threshold);
    if (cluster_method == InitMethodImport && cluster_file == 0) usage(argv[0], threshold);

    if(is_output_timing) io_start = clock();

    samples = file_read(isBinaryFile, filename, &numSamples, &numFeatures);
    if(samples == NULL) exit(1);
#if optimLevel==1
    gpuErrchk(cudaMallocHost((void**)&membership, numSamples *
                sizeof(uint32_t)));
#else
    membership = (uint32_t *) malloc(numSamples * sizeof(uint32_t));
#endif
    assert(membership != NULL);

    memset(membership, 255, numSamples
             * sizeof(uint32_t));

#if optimLevel==1
    samples_T = transpose(samples[0], numSamples, numFeatures, 1);
#else
    samples_T = transpose(samples[0], numSamples, numFeatures);
#endif
    seed = time(NULL);
    if (cluster_method == InitMethodImport)
    {
        clusters_2d = file_read(isBinaryFile, cluster_file, &numClusters_read,
                &numCluster_Features_read);
        if(clusters_2d == NULL)
        {
            fprintf(stderr, "Invalid cluster file %s\n", cluster_file);
            exit(1);
        }
        if(numClusters_read != numClusters || numCluster_Features_read !=
                numFeatures)
        {
            fprintf(stderr, "Cluster sizes don't match provided inputs\n");
            exit(1);
        }
#if optimLevel==1
        clusters = transpose(clusters_2d[0], numClusters, numFeatures,1);
#else
        clusters = transpose(clusters_2d[0], numClusters, numFeatures);
#endif
        free(clusters_2d[0]);
        free(clusters_2d);
    }
    else
    {
#if optimLevel==1
        gpuErrchk(cudaMallocHost((void**) &clusters, numClusters * numFeatures
                    *sizeof(float)));
#else
        clusters = (float *) malloc(numClusters * numFeatures * sizeof(float));
#endif
    }
    if(is_output_timing)
    {
        io_end = clock();
        io_time = ((double)(io_end - io_start)) / CLOCKS_PER_SEC;
        cuda_start = clock();
    }

    //GPU part
    cudaStream_t stream = 0;
#if optimLevel==1
    gpuErrchk(cudaStreamCreate(&stream));
#endif
    gpuErrchk(cudaMalloc((void **) &d_samples, numSamples * numFeatures *
                sizeof(float)));
    gpuErrchk(cudaMalloc((void **) &d_clusters, numClusters * numFeatures *
                sizeof(float)));
    gpuErrchk(cudaMalloc((void **) &d_memberships, numSamples *
                sizeof(uint32_t)));
    gpuErrchk(cudaMemcpyAsync(d_samples, samples_T, numSamples * numFeatures *
                sizeof(float), cudaMemcpyHostToDevice, stream));
    //moving init_centroid here so that in async cpu and gpu can work in
    //parallel
    init_centroids( cluster_method, numSamples, numFeatures, numClusters,
            seed, samples_T, clusters);
    if(_debug)
    {
        printf("init clusters are as follows:\n");
        print2d(clusters,numFeatures,numClusters);
    }
    gpuErrchk(cudaMemcpyAsync(d_clusters, clusters, numClusters * numFeatures *
                sizeof(float), cudaMemcpyHostToDevice, stream));
    if(_debug > 1)
    {
        printf("printing cpu mem\n");
        print1d(membership,numSamples);
    }
    gpuErrchk(cudaMemcpyAsync(d_memberships, membership, numSamples *
                sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    
    if(is_output_timing)
        compute_start = clock();
    kmeans_cuda( InitMethodRandom, threshold, numSamples, numFeatures,
            numClusters, seed, d_samples, d_clusters, d_memberships,
            &numIterations);
    if(is_output_timing)
        compute_end = clock();

    gpuErrchk(cudaMemcpy(clusters, d_clusters, numClusters * numFeatures *
                sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(membership, d_memberships, numSamples *
                sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if(is_output_timing)
    {
        cuda_end = clock();
        cuda_time = ((double)(cuda_end - cuda_start)) / CLOCKS_PER_SEC;
        compute_time = ((double)(compute_end - compute_start)) / CLOCKS_PER_SEC;
        io_start = clock();
    }

    clusters_T = transpose(clusters, numFeatures, numClusters);
    clusters_2d = (float**)malloc(numClusters * sizeof(float*));
    for(uint32_t i = 0; i < numClusters; i++)
        clusters_2d[i] = clusters_T + i * numFeatures;
    if(_debug)
    {
        printf("post cluster centroids are:\n");
        print2d(clusters,numFeatures,numClusters);
    }
    file_write(filename, numClusters, numSamples, numFeatures, clusters_2d,
            membership);
    if(is_output_timing)
    {
        io_end = clock();
        io_time += ((double)(io_end - io_start)) / CLOCKS_PER_SEC;
        printf("I/O time = %10.4f s\n",io_time);
        printf("CUDA time = %10.4f s\n",cuda_time);
        printf("only compute time = %10.4f s\n",compute_time);
    }
    printf("It ran %d number of iterations\n", numIterations);
    free(samples[0]);
    free(samples);
#if optimLevel==1
    gpuErrchk(cudaFreeHost(membership));
    gpuErrchk(cudaFreeHost(samples_T));
    gpuErrchk(cudaFreeHost(clusters));
    gpuErrchk(cudaStreamDestroy(stream));
#else
    free(membership);
    free(samples_T);
    free(clusters);
#endif
    free(clusters_T);
    free(clusters_2d);
    gpuErrchk(cudaFree(d_samples));
    gpuErrchk(cudaFree(d_clusters));
    gpuErrchk(cudaFree(d_memberships));
    return 0;
}
