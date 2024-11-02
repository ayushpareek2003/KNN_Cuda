#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>







namespace knn{
class KNN{
    public:
        KNN(bool is_cudaTrue=true,int distanceType=1,const std::string pathToCSV);
        
        void fit(const std::vector<std::vector<float>> mainData);

        int predict(const std::vector<std::vector<float>> newData);
        
        void transferDataToDevice(float **deviceData,const std::vector<std::vector<float>>
                                                                         hostData);
        




    private:
        bool is_cudaTrue;
        int distanceType;
        std::string pathToCSV;
        std::vector<std::vector<float>> mainData;
        float** deviceData;


    
};


}

