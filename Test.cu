#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
__global__ void KNN_CUDA(float *deviceData, float *testData, int rows, int cols, float *distances) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < rows*cols) {
        int row = tid / cols; 
        int col = tid % cols; 
        
        float diff = deviceData[tid] - testData[col]; 
        
        atomicAdd(&distances[row], diff * diff); 

    }
}




int main(){
    float arr[] = {5.0, 5.8, 4.6, 1.2, 1.5,7.8};

    float *d;
    cudaMalloc(&d,6*sizeof(float));
    cudaMemcpy(d,arr,6*sizeof(float),cudaMemcpyHostToDevice);

    float tes[]={5.0,6.7};

    float *de;
    cudaMalloc(&de,2*sizeof(float));
    cudaMemcpy(de,tes,2*sizeof(float),cudaMemcpyHostToDevice);

    float *dis;
    cudaMalloc(&dis,3*sizeof(float));

    KNN_CUDA<<<1,128>>>(d,de,3,2,dis);

    float *arr2;
    arr2=(float*)malloc(3*sizeof(float));
    cudaMemcpy(arr2,dis,3*sizeof(float),cudaMemcpyDeviceToHost);

    for(int l=0;l<3;l++){
        std::cout<<arr2[l]<<" ";
    }

    return 0;







}