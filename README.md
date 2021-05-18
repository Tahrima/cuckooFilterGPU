# cuckooFilterGPU

A parallel implementation of a Cuckoo Filter described in:

Fan, Bin, et al. "Cuckoo filter: Practically better than bloom." Proceedings of the 10th ACM International on Conference on emerging Networking Experiments and Technologies. ACM, 2014.

Created in Rutgers University Compilers 415

**Note: This implementation is provided as-is**


## Cuckoo Filter Summary
A Cuckoo filter is a Approximate Membership Query structure that is an improvement upon Bloom Filters. Unlike Bloom Filters, Cuckoo Filters provide deletion capability with less space cost than a Quotient Filter.


### Structure
The cuckoo filter is an array of k-sized buckets. Each item that is inserted into the filter is hashed to a fingerprint, which is then inserted into the bucket. This fingerprint is used to identify the item later on lookup.

Each fingerprint can be placed in one of two buckets:
1. `hash(x)`
2. `hash(x) XOR hash(fingerprint)`

The location within the bucket does not matter. If both of the buckets are full, a fingerprint from one of the buckets is removed(or "kicked back") and placed in its alternative bucket. This continues until all fingerprints are successfully placed in a location.



## Parallel Algorithm

### Batch Insertion Pre-proccessing
In our parallel implementation, we aimed to solve the issue of recursive kickbacks that the sequential insertion has.

When inserting a large number of items at once, we visualized this process as a graph where:

```
G(V,E) is a graph where
    V is a set of vertices, each of which represents a bucket in the array
    E is a set of directed edges, each represents an item x. Each edge has:
        a weight which is fingerprint of the item
        a source which is the hash(x)
        a destination, which is the hash(x) XOR hash(fingerprint)
        a bit flag, which indicates the direction of the edge
            0: source -> destination
            1: source <- destination
```

We populate the graph assuming each item `x` is placed in the bucket index `hash(x)`. The number of indegrees for each vertex indicates the number of items in the bucket, which currently can be more than the bucket size. The parallel algorithm attempts to flip edges direction by changing the bit flag in order to bring the indegree of all vertices below the bucket size. 

Once this is completed, all items can be placed into the resultant Cuckoo Filter based on the vertex each edge is pointing to. 


##### Limitations
Currently the code performs a batch insertion into an empty Cuckoo Filter. It is possible to insert into a non-empty cuckoo filter, if the already inserted items are also inserted into the graph.
