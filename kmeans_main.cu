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
        "       -d             : enable debug mode\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}

int main(int argc, char * argv[])
{
    int opt;
    extern char *optarg;
    extern int optind;
    int isBinaryFile;
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
    
    /* some default values */
    _debug           = 0;
    threshold        = 0.001;
    numClusters      = 0;
    isBinaryFile     = 0;
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

    samples = file_read(isBinaryFile, filename, &numSamples, &numFeatures);
    if(samples == NULL) exit(1);

    membership = (uint32_t *) malloc(numSamples * sizeof(uint32_t));
    assert(membership != NULL);

    memset(membership, 255, numSamples
             * sizeof(uint32_t));

    samples_T = transpose(samples[0], numSamples, numFeatures);
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
        clusters = transpose(clusters_2d[0], numClusters, numFeatures);
        free(clusters_2d[0]);
        free(clusters_2d);
    }
    else
    {
        clusters = (float *) malloc(numClusters * numFeatures * sizeof(float));
    }
    init_centroids( cluster_method, numSamples, numFeatures, numClusters,
            seed, samples_T, clusters);
    if(_debug)
    {
        printf("init clusters are as follows:\n");
        print2d(clusters,numFeatures,numClusters);
    }

    //GPU part
    gpuErrchk(cudaMalloc((void **) &d_samples, numSamples * numFeatures *
                sizeof(float)));
    gpuErrchk(cudaMalloc((void **) &d_clusters, numClusters * numFeatures *
                sizeof(float)));
    gpuErrchk(cudaMalloc((void **) &d_memberships, numSamples *
                sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(d_samples, samples_T, numSamples * numFeatures *
                sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_clusters, clusters, numClusters * numFeatures *
                sizeof(float), cudaMemcpyHostToDevice));
    if(_debug > 1)
    {
        printf("printing cpu mem\n");
        print1d(membership,numSamples);
    }
    gpuErrchk(cudaMemcpy(d_memberships, membership, numSamples *
                sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    kmeans_cuda( InitMethodRandom, threshold, numSamples, numFeatures,
            numClusters, seed, d_samples, d_clusters, d_memberships,
            &numIterations);

    gpuErrchk(cudaMemcpy(clusters, d_clusters, numClusters * numFeatures *
                sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(membership, d_memberships, numSamples *
                sizeof(uint32_t), cudaMemcpyDeviceToHost));

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
    printf("It ran %d number of iterations\n", numIterations);
    return 0;
}
