#include "..\inc\knn_cuda.hpp"
#include <cuda_runtime.h>




__global__ int KNN_CUDA(float **deviceData,int rows,int col,float **deviceDataNew){

    int x=threadIdx.x+blockDim.x*blockIdx.x;

    
    





}

knn::KNN::KNN(bool is_cudaTrue,int distanceType,const std::string pathToCSV):is_cudaTrue(is_cudaTrue),
                                                            distanceType(distanceType),
                                                            pathToCSV(pathToCSV){

    hostData=csvTOvector(pathToCSV);

}



int knn::KNN::predict(const std::vector<std::vector<float>> newData){

    



}

void knn::KNN::fit(){
    float* deviceData;
    if(is_cudaTrue){
        transferDataToDevice();

    }
    else{
        std::cout<<"USE CUDA BHAI"<<std::endl;

        
    }


}

void knn::KNN::transferDataToDevice(){

        int rows=hostData.size();
        int col=0;
        if(rows>0){
            col=hostData[0].size();
        }
        else{
            col=0;
        }
        float* h_data_temp=new float[rows*col];

        for(int i=0;i<rows;i++){
            for(int j=0;j<col;j++){
                h_data_temp[i*(col)+j]=hostData[i][j];

            }
        }

        //cuda time 
        cudaMalloc(deviceData,sizeof(float)*rows*col);
        cudaMemcpy(deviceData,h_data_temp,sizeof(float)*rows*col,cudaMemcpyHostToDevice);

        delete[] h_data_temp;

}

std::vector<std::vector<float>> knn::KNN::csvTOvector(const std::string path){
            std::ifstream file(pathToCSV);
            std::string line;

            std::vector<std::vector<float>> ret;
            if(file.is_open()){
                while(getline(file,line)){
                    std::stringstream ss(line);
                    std::string value;
                    std::vector<float> row;
                    while(getline(ss,value,',')){
                        row.push_back(std::stof(value));
                    }
                    ret.push_back(row);
                }
                file.close();
            }  
            else{
                std::cerr<<"Unable to open file"<<std::endl;
                
            } 

            return ret;
}


