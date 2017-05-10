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
#include "hash/hash_functions.cu"

#define LARGE_THRESHOLD_VAL 10000
#define NUM_BUCKETS 5

__device__ uint64_t TwoIndependentMultiplyShift(uint64_t key) {
    const uint64_t SEED[4] = {0x818c3f78ull, 0x672f4a3aull, 0xabd04d69ull, 0x12b51f95ull};
    const uint64_t m = SEED[0];
    const uint64_t a = SEED[2];
    return (a + m * key) >> 56;
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

    __device__ __host__ Graph(unsigned int max_bucket_size, unsigned int size) {
      num_edges = size;
      for(int i=0; i<NUM_BUCKETS; i++){
        buckets[i] = -max_bucket_size;
      }
      edges = NULL;
    }

    __device__ void printGraph() {
      int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
      if(thread_id == 0) {
        for(int i=0; i<num_edges; i++) {
          printf("Edge %u: %d \t src: %u \t dst: %u\n",i, edges[i].fp, edges[i].src, edges[i].dst);
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
 * @param g is an address in the GPU to place result. Assumes g->edges has been given enough space for @param entryListSize items
 */
__global__ void findAllCollisions(int* entries, int entryListSize, Graph * g) {
  int total_threads = blockDim.x * gridDim.x; //total threads
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
  int thread_id_block = threadIdx.x; //thread number in block


  // CHANGE BELOW LINE TO BE MORE EFFICIENT
  int rounds = entryListSize % total_threads == 0 ? (entryListSize/total_threads):((entryListSize/total_threads)+1);
  g->num_edges = entryListSize;

  for (size_t i = 0; i <rounds; i++) {
    int currIdx = rounds*total_threads + thread_id;
    int * entry = &entries[currIdx];

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
  syncthreads();
  g->printGraph();
}

/**
 * Edge Processing Kernel
 * Finds random edges to evict until capacity for each bucket is equal to 0
 *
 */
__global__ void processEdges(Graph * g, int* anyChange) {
  int total_threads = blockDim.x * gridDim.x; //total threads
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
  int thread_id_block = threadIdx.x; //thread number in block
  int num_edges = g->num_edges;

  int rounds = num_edges % total_threads == 0 ? (num_edges/total_threads):(num_edges/total_threads);

  for(int i=0; i<rounds; i++) {
  	int currIdx = total_threads*i + thread_id; //current edge to process
    Edge *e = &g->edges[currIdx];

    //determine the bucket it's in
    int curr_bucket = e->dir == 0 ? e->src:e->dst;

    //check the bucket
    int * bucketCount = &(g->buckets[curr_bucket]);
    int tmp = *bucketCount;

    //decrement the bucket count if > 0
    if(*bucketCount > 0) {
      printf("flipping direction\n");
      int old = atomicDec((unsigned int *)bucketCount, INT_MAX);
      if (old && old < LARGE_THRESHOLD_VAL) {
      	e->dir = e->dir ^ 1; // flip the bit
        *anyChange = 1;
      }
    }

  }
}

void initGraphCPU(int entry_size) {
	Graph * graph;
  	cudaMalloc(&graph, sizeof(Graph));
    Edge * e;
    cudaMalloc(&e, sizeof(Edge)*entry_size);
}

void insert(int* entries, unsigned int num_entries, unsigned int bucket_size, int num_buckets){
  std::cout << "Inserting " << num_entries << " entries"<< std::endl;
	int anychange = 1;
  	int * d_change = (int *) cudaMallocAndCpy(sizeof(int), &anychange);

  	Graph *h_graph = new Graph(bucket_size, num_entries);

  	//set up pointer
  	cudaMalloc((void**)&(h_graph->edges), sizeof(Edge)*num_entries);
  	Graph *d_graph = (Graph *) cudaMallocAndCpy(sizeof(Graph), h_graph);
  	int * d_entries = (int *) cudaMallocAndCpy(sizeof(int)*num_entries, entries);

  	while (anychange != 0){
      std::cout << "Calling kernel" << std::endl;
      anychange = 0;
      cudaSendToGPU(d_change, &anychange, sizeof(int));

      findAllCollisions<<<2, 512>>>(d_entries, num_entries, d_graph);
      cudaDeviceSynchronize();
      std::cout << "Found all collisions" << std::endl;

      processEdges<<<ceil(num_entries/1024), 1024>>>(d_graph, d_change);
      cudaDeviceSynchronize();
      std::cout << "Proccessed edge" << std::endl;

      cudaGetFromGPU(&anychange, d_change, sizeof(int));
      std::cout << "Got value of anychange: " << anychange << std::endl;

    }
}
