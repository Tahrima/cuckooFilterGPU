#include <stdio.h>
#include <assert.h>
#include <cuda_profiler_api.h>

#include "mt19937ar.h"
#include "CuckooFilter.cu"
#include "graph_test.cu"

#ifndef NOT_FOUND
#define NOT_FOUND UINT_MAX
#endif

void generateRandomNumbers(unsigned int *numberArray, unsigned int n)
{
    init_genrand(time(NULL));   //initialize random number generator
    for (int i = 0; i < n; i++){
        numberArray[i] = genrand_int32();
    }
}

void CUDAErrorCheck()
{
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    errSync = cudaGetLastError();
    errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

int main(int argc, char* argv[])
{

    unsigned int numBuckets = atoi(argv[1]);
    unsigned int bucketSize = atoi(argv[2]);
    float fillFraction = (float)atof(argv[3]);
    unsigned int numLookUps = atoi(argv[4]);
    //New random batch lookups
    //Generate values for random lookups

    int insertSize = floor(numBuckets*bucketSize*fillFraction);
    unsigned int* h_insertValues = new unsigned int[insertSize];
    generateRandomNumbers(h_insertValues, insertSize);

    CuckooFilter * ckFilter = new CuckooFilter(numBuckets, bucketSize);
    insert((int *)h_insertValues, insertSize, numBuckets, bucketSize, ckFilter);

    unsigned int* h_lookUpValues = new unsigned int[numLookUps];
    generateRandomNumbers(h_insertValues, numLookUps);

    unsigned int * d_lookUpValues;
    cudaMalloc((void**) &d_lookUpValues, numLookUps * sizeof(unsigned int));
    cudaMemcpy(&d_lookUpValues, &h_lookUpValues, numLookUps * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Output array
    char * d_results;
    cudaMalloc((void**) &d_results, numLookUps * sizeof(char));
    cudaMemset(&d_results, 0, numLookUps * sizeof(char));
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    //Launch lookup kernel
    // cudaProfilerStart();
    // cudaEventRecord(start);
    lookUpGPU<<<(numLookUps + 1023)/1024, 1024>>>(ckFilter, numLookUps, d_lookUpValues, d_results);
    cudaDeviceSynchronize();
    char * h_results = new char[numLookUps];
    cudaMemcpy(&h_results, &d_results, numLookUps* sizeof(char), cudaMemcpyDeviceToHost);
    // cudaEventRecord(stop);
    // cudaProfilerStop();

    //Calculate and print timing results
    // cudaEventSynchronize(stop);
    // float batchLookupTime = 0;
    // cudaEventElapsedTime(&batchLookupTime, start, stop);
//    printf("Random lookup rate = %f million ops/sec\n", numValues / randomLookupTime / 1000);
    //printf("%f\n", batchSize / batchLookupTime / 1000);

    //Free Memory
    ckFilter->freeFilter();
    delete[] h_insertValues;
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    delete[] h_lookUpValues;
    cudaFree(d_lookUpValues);
    cudaFree(d_results);
    delete[] h_results;
    cudaDeviceReset();

    return 0;
}
