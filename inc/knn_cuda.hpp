#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>







namespace knn{
class KNN{
    public:
        KNN(bool is_cudaTrue=true,int distanceType=1,const std::string pathToCSV);
        
        void fit();

        int predict(const std::string testData);
        
        void transferDataToDevice();

        std::vector<std::vector<float>> csvTOvector(const std::string path);

        __global__ void KNN_CUDA(float **deviceData,int rows,int cols,float *distances);



    private:
        bool is_cudaTrue;
        int distanceType;
        std::string pathToCSV;
        std::vector<std::vector<float>> hostData;
        float** deviceData;
        int rows;
        int cols;

    
};


}

