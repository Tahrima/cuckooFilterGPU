//GPUBatchInserts.cu

#include <stdio.h>
#include <assert.h>
#include <cuda_profiler_api.h>

#include "../mt19937ar.h"
#include "quotientFilter.cuh"

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
    assert(argc == 5);
    int q = atoi(argv[1]);
    int r = atoi(argv[2]);
    float alpha = atof(argv[3]);    //initial fill %
    int batchSize = atoi(argv[4]);  //size of batch to insert after build

    //Initialize filter
    struct quotient_filter d_qfilter;
    initFilterGPU(&d_qfilter, q, r);
    cudaMemset(d_qfilter.table, 0, calcNumSlotsGPU(q, r) * sizeof(unsigned char));
    
    //Generate set of random numbers
    int numValues = alpha * (1 << q);
    unsigned int* h_randomValues = new unsigned int[numValues];
    generateRandomNumbers(h_randomValues, numValues);
    unsigned int* d_randomValues;
    cudaMalloc((void**) &d_randomValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_randomValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

//Random Inserts
    float filterBuildTime = insert(d_qfilter, numValues, d_randomValues, &d_qfilter.cardinality);
//    printf("Insert rate = %f million ops/sec\n", numValues / filterBuildTime / 1000);

//Insert new batch
    unsigned int* h_newBatch = new unsigned int[batchSize];
    generateRandomNumbers(h_newBatch, batchSize);
    unsigned int* d_newBatch;
    cudaMalloc((void**) &d_newBatch, batchSize * sizeof(unsigned int));
    cudaMemcpy(d_newBatch, h_newBatch, batchSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    float insertTime = insert(d_qfilter, batchSize, d_newBatch, &d_qfilter.cardinality);
    printf("%f\n", batchSize / insertTime / 1000);

    //Free Memory
    cudaFree(d_qfilter.table);
    delete[] h_randomValues;
    cudaFree(d_randomValues);
    delete[] h_newBatch;
    cudaFree(d_newBatch);
    cudaDeviceReset();

    return 0;
}
