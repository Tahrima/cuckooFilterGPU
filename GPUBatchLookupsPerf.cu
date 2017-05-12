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

    assert(argc==4);
    unsigned int numBuckets = atoi(argv[1]);
    unsigned int bucketSize = atoi(argv[2]);
    float fillFraction = (float)atof(argv[3]);
    //Generate values for random lookups

    int insertSize = floor(numBuckets*bucketSize*fillFraction);
    unsigned int* h_insertValues = new unsigned int[insertSize];
    generateRandomNumbers(h_insertValues, insertSize);

    CuckooFilter * ckFilter = new CuckooFilter(numBuckets, bucketSize);
    insert((int *)h_insertValues, insertSize, numBuckets, bucketSize, ckFilter);

    // Lookup values are the inserted values from earlier.
    unsigned int * d_lookUpValues;
    cudaMalloc((void**) &d_lookUpValues, insertSize * sizeof(unsigned int));
    cudaMemcpy(d_lookUpValues, h_insertValues, insertSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Output array
    char * d_results;
    cudaMalloc((void**) &d_results, insertSize * sizeof(char));
    cudaMemset(&d_results, 0, insertSize * sizeof(char));

    CuckooFilter * d_ckFilter = (CuckooFilter *) cudaMallocAndCpy(sizeof(CuckooFilter), ckFilter);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Launch lookup kernel
    cudaProfilerStart();
    cudaEventRecord(start);

    std::cout << "Calling lookup kernel" << std::endl;
    lookUpGPU<<<(insertSize + 1023)/1024, 1024>>>(d_ckFilter, insertSize, d_lookUpValues, d_results);
    cudaDeviceSynchronize();
    char * h_results = new char[insertSize];
    cudaMemcpy(h_results, d_results, insertSize* sizeof(char), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaProfilerStop();

    //Calculate and print timing results
    cudaEventSynchronize(stop);
    float batchLookupTime = 0;
    cudaEventElapsedTime(&batchLookupTime, start, stop);
    printf("%f\n", insertSize / batchLookupTime / 1000);
    //Free Memory
     delete[] h_insertValues;
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    cudaFree(d_lookUpValues);
    cudaFree(d_results);
    delete[] h_results;
    cudaDeviceReset();

    return 0;
}
