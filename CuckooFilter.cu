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
      for (int i = 0; i < numBuckets; i++) {
        cudaFree(buckets[i]);
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
