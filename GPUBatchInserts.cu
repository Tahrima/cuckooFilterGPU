//GPUBatchInserts.cu

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
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n; i++){
        numberArray[i] = rand();
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
    // assert(argc == 5);
    // int q = atoi(argv[1]);
    // int r = atoi(argv[2]);
    // float alpha = atof(argv[3]);    //initial fill %
    // int batchSize = atoi(argv[4]);  //size of batch to insert after build

    //TODO: Initialize filter
    /*struct quotient_filter d_qfilter;
    initFilterGPU(&d_qfilter, q, r);
    cudaMemset(d_qfilter.table, 0, calcNumSlotsGPU(q, r) * sizeof(unsigned char));
    */
    //Generate set of random numbers
    assert(argc==4);
    unsigned int numBuckets = atoi(argv[1]);
    unsigned int bucketSize = atoi(argv[2]);
    float fillFraction = (float)atof(argv[3]);

    int insertSize = floor(numBuckets*bucketSize*fillFraction);
    unsigned int* h_randomValues = new unsigned int[insertSize];
    generateRandomNumbers(h_randomValues, insertSize);

    CuckooFilter * ckFilter = new CuckooFilter(numBuckets, bucketSize);
    insert((int *)h_randomValues, insertSize, numBuckets, bucketSize, ckFilter);
//    printf("Insert rate = %f million ops/sec\n", numValues / filterBuildTime / 1000);


//TODO: Insert new batch

    //Free Memory
    delete[] h_randomValues;
    CUDAErrorCheck();
    cudaDeviceReset();

    return 0;
}
