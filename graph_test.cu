/*
 * Parallel Graph Preprocessing of cuckoo filter
 * This preprocesses a batch insertion into a cuckoo filter by creating a directed graph (V,E) where:
 *    V is a set of vertices that represent each bucket of the cuckoo filter
 *    E is a set of edges (u,v) with weight w where:
 *      w is the fingerprint of a specific entry
 *      u is the bucket number given by hash(entry)
 *      v is the bucket number given by hash(entry) xor hash(fingerprint)
 *      dir indicates the vertex pointed to by the edge. Also indicates
 *          which bucket number the fingerprint should be placed in.
 */

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

#define LARGE_THRESHOLD_VAL 10000
#define NUM_BUCKETS 100
#define MAX_BUCKET_SIZE 4
__device__ uint64_t TwoIndependentMultiplyShift(unsigned int key) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
    const uint64_t SEED[4] = {0x818c3f78ull, 0x672f4a3aull, 0xabd04d69ull, 0x12b51f95ull};
    const uint64_t m = SEED[(thread_id %2)+2];
    const uint64_t a = SEED[thread_id % 2];
    //printf("thread: %d \t key: %u, m: %u, a: %u = %lu\n",thread_id, key, m, a, (a + m * key));
    return (a + m * key);
}
__device__ void random(unsigned int seed, int* result, int max) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* we have to initialize the state */
  curand_init(seed, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  /* curand works like rand - except that it takes a state as a parameter */
  *result = curand(&state) % max;
}
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}

void * cudaMallocAndCpy(int size, void * hostMemory) {
  void * gpuMem;
  cudaMalloc((void**) &gpuMem, size);
  if (hostMemory != NULL) {
    cudaMemcpy(gpuMem, hostMemory, size, cudaMemcpyHostToDevice);
  }
  return gpuMem;
}

void cudaGetFromGPU(void * destination, void * gpuMemory, int size) {
  cudaMemcpy(destination, gpuMemory, size, cudaMemcpyDeviceToHost);
}

void cudaSendToGPU(void * destination, void * hostMemory, int size) {
  cudaMemcpy(destination, hostMemory, size, cudaMemcpyHostToDevice);
}

class Edge {
  public:
    unsigned int src; //hash(x) location
    unsigned int dst; //hash(x) xor hash(fp) location
    unsigned char fp; //fingerprint
    int dir; //0 to be src, 1 to be dst

 	__device__ __host__ Edge(){}
};

class Graph {
  public:
    int buckets[NUM_BUCKETS]; //value at index i is the number of indegrees to a bucket i
  	Edge *edges;
  	unsigned int num_edges;
    __device__ __host__ Graph(unsigned int size) {
      num_edges = size;
      for(int i=0; i<NUM_BUCKETS; i++){
        buckets[i] = 0;
      }
      edges = NULL;
    }

    __device__ void printGraph() {
      int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
      if(thread_id == 0) {
        for(int i=0; i<num_edges; i++) {
          printf("Edge %d: %u \t src: %u \t dst: %u\n",i, edges[i].fp, edges[i].src, edges[i].dst);
        }
        printCollisions();

      }
    }

    __device__ void printCollisions() {
      int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
      if(thread_id == 0) {
        printf("\n\nBuckets\n");
        for(int i=0; i<NUM_BUCKETS; i++) {
          if(buckets[i] > MAX_BUCKET_SIZE) {
            printf("Collisions for bucket %d: %d\n", i, buckets[i]);
          }
        }
      }
    }
};

// __global__ void setup_kernel (curandState * state, Graph *g)
// {
//   	int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
//    	// change sequence number to currIdx if values are too correlated
//   	// curand_init(1234, 0, 0, &state[currIdx]);
//     curand_init(1234, 0, 0, &state[currIdx]);

// }


/**
 * Parallel graph building
 * @param entries is a list of entries to enter
 * @param entryListSize is the size of the @param entries list
 * @param g is an address in the GPU to pla\ce result. Assumes g->edges has been given enough space for @param entryListSize items
 */
__global__ void findAllCollisions(int* entries, int entryListSize, Graph * g) {
  int total_threads = blockDim.x * gridDim.x; //total threads
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
  int thread_id_block = threadIdx.x; //thread number in block

  // CHANGE BELOW LINE TO BE MORE EFFICIENT
  int rounds = entryListSize % total_threads == 0 ? (entryListSize/total_threads):((entryListSize/total_threads)+1);
  g->num_edges = entryListSize;

  for (size_t i = 0; i <rounds; i++) {
    int currIdx = i*total_threads + thread_id;
    if(currIdx < entryListSize) {
      int * entry = &entries[currIdx];

      //printf("KERNEL SPACE current Index %d, Thread id %d: %x\n", currIdx, thread_id, entry);
      unsigned int bucket1;
      hash_item((unsigned char*) entry,
                    4,
                    NUM_BUCKETS,
      		      HASHFUN_NORM,
                    &bucket1);

      const uint64_t hash = TwoIndependentMultiplyShift(*entry);
      unsigned char fp = (unsigned char) hash;
      unsigned int fpHash;
      hash_item((unsigned char*) &fp,
                    1,
                    NUM_BUCKETS,
      		      HASHFUN_NORM,
                    &fpHash);
      unsigned int bucket2 = (bucket1 ^ fpHash) & 0b11111111;

      //build edge

      g->edges[currIdx].fp = fp;
      g->edges[currIdx].src = bucket1 % NUM_BUCKETS;
      g->edges[currIdx].dst = bucket2 % NUM_BUCKETS;


  // 	Copy state to local memory for efficiency */
  //     curandState local_state = global_state[thread_id];
  // 	/* Generate pseudo - random unsigned ints
  //     g->edges[i].dir = curand_uniform(&local_state);

      //update bucket
      atomicAdd(&(g->buckets[bucket1]), 1);
    }
  }
  syncthreads();
}
__global__ void resetCollisions(Graph * g) {


  g->printCollisions();
  int total_threads = blockDim.x * gridDim.x; //total threads
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
  int thread_id_block = threadIdx.x; //thread number in block

  int rounds = (NUM_BUCKETS % total_threads == 0) ? (NUM_BUCKETS/total_threads):(NUM_BUCKETS/total_threads + 1);

  for (size_t iter = 0; iter < rounds; iter++) {
    int currIdx = iter*total_threads + thread_id;
    if(currIdx < NUM_BUCKETS) {
      int * currBucket = &(g->buckets[currIdx]);
      *currBucket = 0;
    }

  }

  rounds = (g->num_edges % total_threads == 0) ? (g->num_edges/total_threads):(g->num_edges/total_threads + 1);

  for (size_t iter = 0; iter < rounds; iter++) {
    int currIdx = iter*total_threads + thread_id;

    if(currIdx < g->num_edges) {
      int b = (g->edges[currIdx].dir == 0) ? (g->edges[currIdx].src):(g->edges[currIdx].dst);
      atomicAdd(&(g->buckets[b]),1);
    }
  }
  g->printCollisions();
}

/**
 * Edge Processing Kernel
 * Finds random edges to evict until capacity for each bucket is equal to 0
 *
 */
__global__ void processEdges(Graph * g, int* anyChange, unsigned int randNum) {
  int total_threads = blockDim.x * gridDim.x; //total threads
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
  int thread_id_block = threadIdx.x; //thread number in block
  int num_edges = g->num_edges;

  int rounds = num_edges % total_threads == 0 ? (num_edges/total_threads):(num_edges/total_threads+1);

  for(int i=0; i<rounds; i++) {
  	int currIdx = total_threads*i + thread_id; //current edge to process
    if(currIdx < g->num_edges) {
      Edge *e = &g->edges[currIdx];

      //determine the bucket it's in
      int curr_bucket = e->dir == 0 ? e->src:e->dst;

      //check the bucket
      int * bucketCount = &(g->buckets[curr_bucket]);
      int tmp = *bucketCount;
      //decrement the bucket count if > 0
      //int rand;
      //random((unsigned int)clock() + thread_id, &rand, 50);
      syncthreads();
      if(*bucketCount > MAX_BUCKET_SIZE) {
        int old = atomicDec((unsigned int *)bucketCount, INT_MAX);
        int shift = randNum % tmp;
        int shiftedValue = old - shift;
        int bucketOffset = (shiftedValue < 0) ? tmp - abs(shiftedValue) : shiftedValue;
        if (bucketOffset > MAX_BUCKET_SIZE && old < LARGE_THRESHOLD_VAL) {
          if (e->dir) {
            printf("tmp %d, old %d, shift %d, shiftedValue %d, bucketOffset %d \t Evicting %d from %d to %d\n", tmp, old, shift, shiftedValue, bucketOffset, e->fp, e->dst, e->src);
          } else {
            printf("tmp %d, old %d, shift %d, shiftedValue %d, bucketOffset %d \t Evicting %d from %d to %d\n", tmp, old, shift, shiftedValue, bucketOffset, e->fp, e->src, e->dst);
          }
        	e->dir = e->dir ^ 1; // flip the bit
          *anyChange = 1;
        }
      }
    }
  }

  __syncthreads();
  g->printCollisions();
}

void initGraphCPU(int entry_size) {
	Graph * graph;
  	cudaMalloc(&graph, sizeof(Graph));
    Edge * e;
    cudaMalloc(&e, sizeof(Edge)*entry_size);
}

void insert(int* entries, unsigned int num_entries){
  std::cout << "Inserting " << num_entries << " entries"<< std::endl;
	int anychange = 1;
  	int * d_change = (int *) cudaMallocAndCpy(sizeof(int), &anychange);

  	Graph *h_graph = new Graph(num_entries);

  	//set up pointer
  	cudaMalloc((void**)&(h_graph->edges), sizeof(Edge)*num_entries);
  	Graph *d_graph = (Graph *) cudaMallocAndCpy(sizeof(Graph), h_graph);
  	int * d_entries = (int *) cudaMallocAndCpy(sizeof(int)*num_entries, entries);

    std::cout << "Calling kernel" << std::endl;
    findAllCollisions<<<2, 512>>>(d_entries, num_entries, d_graph);
    cudaDeviceSynchronize();
    int count = 0;
  	while (anychange != 0){
      anychange = 0;
      cudaSendToGPU(d_change, &anychange, sizeof(int));

      // generate random number

      unsigned int randNum = rand() % (NUM_BUCKETS * 8);
      std::cout << "Found all collisions, rand num: "<< randNum << std::endl;
      processEdges<<<ceil((double)num_entries/1024), 1024>>>(d_graph, d_change, randNum);
      cudaDeviceSynchronize();
      std::cout << "Proccessed edge using " << ceil((double)num_entries/1024) << "threads " << std::endl;

      cudaGetFromGPU(&anychange, d_change, sizeof(int));
      std::cout << "Got value of anychange: " << anychange << std::endl;
      if(anychange == 1){
        resetCollisions<<<ceil((double)num_entries/1024), 1024>>>(d_graph);
      }
      cudaDeviceSynchronize();
      count++;
    }
    printf("Count: %d\n",count);
}
