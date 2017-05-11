#include <stdio.h>
#include <assert.h>
#include <cuda_profiler_api.h>

#include "../mt19937ar.h"

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
    unsigned int numValues = atoi(argv[3]);
    //New random batch lookups
    //Generate values for random lookups
    unsigned int* h_batchLookupValues = new unsigned int[batchSize];
    generateRandomNumbers(h_batchLookupValues, batchSize);

    //Array of lookup values
    unsigned int* d_batchLookupValues;
    cudaMalloc((void**) &d_batchLookupValues, batchSize * sizeof(int));
    cudaMemcpy(d_batchLookupValues, h_batchLookupValues, batchSize * sizeof(int), cudaMemcpyHostToDevice);

    //Output array
    unsigned int* d_batchReturnValues;
    cudaMalloc((void**) &d_batchReturnValues, batchSize * sizeof(unsigned int));
    cudaMemset(&d_batchReturnValues, 0, batchSize * sizeof(unsigned int));


    CuckooFilter ckFilter = new CuckooFilter(numBuckets, bucketSize);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    //Launch lookup kernel
    // cudaProfilerStart();
    // cudaEventRecord(start);
    lookUp<<<(batchSize + 1023)/1024, 1024>>>(batchSize, d_qfilter, d_batchLookupValues, d_batchReturnValues);
    // cudaEventRecord(stop);
    // cudaProfilerStop();

    //Calculate and print timing results
    // cudaEventSynchronize(stop);
    // float batchLookupTime = 0;
    // cudaEventElapsedTime(&batchLookupTime, start, stop);
//    printf("Random lookup rate = %f million ops/sec\n", numValues / randomLookupTime / 1000);
    //printf("%f\n", batchSize / batchLookupTime / 1000);

    //Free Memory
    cudaFree(d_qfilter.table);
    delete[] h_randomValues;
    cudaFree(d_randomValues);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_batchLookupValues;
    cudaFree(d_batchLookupValues);
    cudaFree(d_batchReturnValues);
    cudaDeviceReset();

    return 0;
}
