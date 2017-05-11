class CuckooFilter {
  public:
    char** buckets;
    unsigned int numBuckets;
    __host__ CuckooFilter(unsigned int numberOfBuckets, unsigned int bucketSize) {
      numBuckets = numberOfBuckets;
      cudaMalloc((void**)&buckets, sizeof(char*) * numBuckets);
      for(int i=0; i<numBuckets; i++){
        cudaMalloc((void**)&buckets[i], sizeof(char) * bucketSize);
      }
    }
    __host__ void freeFilter() {
      for (int i = 0; i < numBuckets; i++) {
        cudaFree(buckets[i]);
      }
      cudaFree(buckets);
    }
    __device__ void insert(unsigned int fingerprint, unsigned int bucketNum, unsigned int index) {
      buckets[bucketNum][index] = fingerprint;
    }

    __device__ void printFilter() {
      int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
      if(thread_id == 0) {
        for(int i=0; i<numBuckets; i++) {
          printf("Bucket %d: \t",i);
          for (int j = 0; j < bucketSize; j++) {
            printf(" | %d |", buckets[i][j]);
          }
          printf("\n");
        }
      }
    }
};
