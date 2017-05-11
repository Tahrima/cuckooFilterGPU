#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <climits>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include "hash/hash_functions.cu"

__device__ uint64_t TwoIndependentMultiplyShift(unsigned int key) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
    const uint64_t SEED[4] = {0x818c3f78ull, 0x672f4a3aull, 0xabd04d69ull, 0x12b51f95ull};
    const uint64_t m = SEED[(thread_id %2)+2];
    const uint64_t a = SEED[thread_id % 2];
    //printf("thread: %d \t key: %u, m: %u, a: %u = %lu\n",thread_id, key, m, a, (a + m * key));
    return (a + m * key);
}

class CuckooFilter {
  public:
    char** buckets;
    unsigned int numBuckets;
    unsigned int bucketSize;
    __host__ CuckooFilter(unsigned int numberOfBuckets, unsigned int bucketSizeIn) {
      numBuckets = numberOfBuckets;
      bucketSize = bucketSizeIn;



      char ** tmpbuckets = new char*[numberOfBuckets];
      for(int i=0; i<numBuckets; i++){
        cudaMalloc((void**)&tmpbuckets[i], sizeof(char) * bucketSize);
        cudaMemset((tmpbuckets[i]), 0, sizeof(char) * bucketSize);
      }
      cudaMalloc((void**)&buckets, sizeof(char*)*numberOfBuckets);
      cudaMemcpy(buckets, tmpbuckets, sizeof(char*)*numberOfBuckets, cudaMemcpyHostToDevice);
    }
    __host__ void freeFilter() {
      char ** tmpBuckets = new char*[bucketSize];
      cudaMemcpy(tmpBuckets, tmpBuckets, sizeof(char*)*numBuckets, cudaMemcpyDeviceToHost);
      for (int i = 0; i < numBuckets; i++) {
        cudaFree(tmpBuckets[i]);
      }
      cudaFree(buckets);
    }
    __device__ void insert(unsigned int fingerprint, unsigned int bucketNum, unsigned int index) {
      buckets[bucketNum][index] = (char)fingerprint;
    }
    __device__ unsigned int lookup(unsigned int bucketNum, unsigned int index) {
      return(buckets[bucketNum][index]);
    }
    __device__ unsigned int lookupFingerprintInBucket(unsigned int fingerprint, unsigned int bucketNum) {
      int retVal = 0;
      for (int i = 0; i < bucketSize; i++) {
        retVal = retVal || (fingerprint == buckets[bucketNum][i]);
      }
      return(retVal);
    }

    __device__ void printFilter() {
      int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
      if(thread_id == 0) {
        for(int i=0; i<numBuckets; i++) {
          printf("Bucket %d: \t",i);
          for (int j = 0; j < bucketSize; j++) {
            printf(" | %u |", (unsigned char)buckets[i][j]);
          }
          printf("\n");
        }
      }
    }
};

__global__ void lookUpGPU(CuckooFilter *ck, int numLookUps, unsigned int* lookUps, char * results){

    int total_threads = blockDim.x * gridDim.x; //total threads
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
    int rounds = (numLookUps % total_threads == 0) ? (numLookUps/total_threads):((numLookUps/total_threads)+1);

    for (size_t i = 0; i < rounds; i++) {
      int currIdx = total_threads*i + thread_id;
      if(currIdx < numLookUps){

        int entry = lookUps[currIdx];
        unsigned int bucket1;
        hash_item((unsigned char*) &entry,
                      4,
                      ck->numBuckets,
                      HASHFUN_NORM,
                      &bucket1);

        const uint64_t hash = TwoIndependentMultiplyShift(entry);
        unsigned char fp = (unsigned char) hash;
        unsigned int fpHash;
        hash_item((unsigned char*) &fp,
                      1,
                      ck->numBuckets,
                      HASHFUN_NORM,
                      &fpHash);
        unsigned int bucket2 = (bucket1 ^ fpHash) & 0b11111111;

        int in_b1 = ck->lookupFingerprintInBucket(fp, bucket1);
        int in_b2 = ck->lookupFingerprintInBucket(fp, bucket2);

        results[currIdx] = in_b1 || in_b2;
      }
    }
    __syncthreads();
}
