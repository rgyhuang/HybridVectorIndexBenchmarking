#include <iostream>
#include <vector>
#include <faiss/IndexACORN.h>

int main() {
    int d = 64;
    int M = 16;
    int gamma = 4;
    int M_beta = 16;
    int N = 10;
    
    // Create metadata vector - MUST be same size as number of vectors that will be added
    std::vector<int> metadata(N);
    for (int i = 0; i < N; i++) {
        metadata[i] = i % gamma;
    }
    
    std::cout << "Creating ACORN index..." << std::endl;
    std::cout << "Metadata size: " << metadata.size() << std::endl;
    faiss::IndexACORNFlat index(d, M, gamma, metadata, M_beta);
    std::cout << "Index created: ntotal=" << index.ntotal << std::endl;
    
    // Create some vectors
    std::vector<float> vectors(N * d);
    for (int i = 0; i < N * d; i++) {
        vectors[i] = (float)rand() / RAND_MAX;
    }
    
    std::cout << "Adding " << N << " vectors..." << std::endl;
    index.add(N, vectors.data());
    std::cout << "Vectors added: ntotal=" << index.ntotal << std::endl;
    
    return 0;
}