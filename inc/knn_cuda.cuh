#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <algorithm>
#include <map>
#include <stdio.h>
namespace knn{
class KNN{
    public:
        KNN(const std::string pathToCSV,int k_NEIG,bool is_cudaTrue=true,int distanceType=1);
        
        void fit();

        int predict(const std::string testData);
        
        void transferDataToDevice();

        std::vector<std::vector<float>> csvTOvector(const std::string path);

      

        int majorityCOUNT(float* distances);



    private:
        bool is_cudaTrue;
        int distanceType;
        std::string pathToCSV;
        std::vector<std::vector<float>> hostData;
        std::vector<int> labels;
        float* deviceData;
        int rows;
        int cols;
        int k_NEIG;

};

}

