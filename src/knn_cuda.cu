#include "..\inc\knn_cuda.hpp"



__constant__ float* newData[128];

__global__ void knn::KNN::KNN_CUDA(float **deviceData,int rows,int cols,float *distances){


    int r=threadIdx.x;
    int c=blockDim.x*blockIdx.x;

    __shared__ float distancesTEMPblock[128];

    if(r+c<rows*cols){
        float* a_TEMP=newData[r];
        float* b_TEMP=deviceData[r+c];
        distancesTEMPblock[r/cols]+=(a_TEMP-b_TEMP)*(a_TEMP-b_TEMP);
        __syncthreads();
    }

    distances[r/cols]=distancesTEMPblock[r/cols];
    

}

knn::KNN::KNN(bool is_cudaTrue,int distanceType,const std::string pathToCSV):is_cudaTrue(is_cudaTrue),
                                                            distanceType(distanceType),
                                                            pathToCSV(pathToCSV){

    hostData=csvTOvector(pathToCSV);

}



int knn::KNN::predict(const std::string testData){

    std::vector<std::vector<float>> readyTestData=csvTOvector(testData);

    int row=readyTestData.size();
    int col=0;
    if(row>0){
            col=readyTestData[0].size();
    }
    else if(row>1){
        return 0;
            col=0;
    }
    float* h_data_temp=new float[row*col];

    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            h_data_temp[i*(col)+j]=hostData[i][j];

        }
    }

    cudaMemcpyToSymbol(newData,h_data_temp,sizeof(h_data_temp));

    float *distances;
    distances=(float*)malloc(sizeof(float)*rows);

    dim3 blocks=(ceil((row*col)/128));
    dim3 threads=(128);

    knn::KNN::KNN_CUDA<<<blocks,threads>>>(deviceData,rows,cols,distances);

    //code to answer that soon i will write that// 











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

        rows=hostData.size();
        cols=0;
        if(rows>0){
            cols=hostData[0].size();
        }
        else{
            cols=0;
        }
        float* h_data_temp=new float[rows*cols];

        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                h_data_temp[i*(cols)+j]=hostData[i][j];

            }
        }

        //cuda time 
        
        cudaMalloc(deviceData,sizeof(float)*rows*cols);
        cudaMemcpy(deviceData,h_data_temp,sizeof(float)*rows*cols,cudaMemcpyHostToDevice);

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


