//GPUBatchInserts.cu

#include <stdio.h>
#include <assert.h>
#include <cuda_profiler_api.h>

#include "mt19937ar.h"
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
    unsigned int numValues = 5;
    int* h_randomValues = new int[numValues];
    // generateRandomNumbers((unsigned int *)h_randomValues, numValues);


    for (size_t i = 0; i < numValues; i++) {
      h_randomValues[i] = i;
      std::cout <<"Number " << i << ": " <<h_randomValues[i] << std::endl;
    }
    // return;
//Random Inserts
    insert(h_randomValues, numValues, 4, 100);
//    printf("Insert rate = %f million ops/sec\n", numValues / filterBuildTime / 1000);


//TODO: Insert new batch

    //Free Memory
    delete[] h_randomValues;
    CUDAErrorCheck();
    cudaDeviceReset();

    return 0;
}
