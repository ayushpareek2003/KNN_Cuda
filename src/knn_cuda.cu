#include "..\inc\knn_cuda.hpp"

//////////////////////////////////////////////////
////////// This is more optimised code ///////////
//////////////////////////////////////////////////


// __constant__ float* newData[128];

// __global__ void KNN_CUDA(float **deviceData,int rows,int cols,float *distances){


//     int r=threadIdx.x;
//     int c=blockDim.x*blockIdx.x;

//     __shared__ float distancesTEMPblock[128];

//     if(r+c<rows*cols){
//         float* a_TEMP=newData[r];
//         float* b_TEMP=deviceData[r+c];
//         distancesTEMPblock[r/cols]+=(a_TEMP-b_TEMP)*(a_TEMP-b_TEMP);
//         __syncthreads();
//     }

//        if (r < 128 && c < rows) {
//         distances[r/cols] = sqrtf(distancesTEMPblock[r/cols]);  // Store the result in distances
//     }

// }

//////////////////////////////////////////////////
///////////i will make it work one day ///////////
//////////////////////////////////////////////////



/// temporary function 
__global__ void KNN_CUDA(float *deviceData, float *testData, int rows, int cols, float *distances) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows) {
        float dist = 0.0;
        for (int j = 0; j < cols; ++j) {
            float diff = testData[j] - deviceData[tid * cols + j];
            dist += diff * diff;
        }
        distances[tid] = sqrtf(dist);
    }
}

knn::KNN::KNN(const std::string pathToCSV,int k_NEIG,bool is_cudaTrue,int distanceType):is_cudaTrue(is_cudaTrue),
                                                            distanceType(distanceType),
                                                            pathToCSV(pathToCSV),k_NEIG(k_NEIG){

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

    float *newData;

    // cudaMemcpyToSymbol(newData,h_data_temp,sizeof(h_data_temp));
    cudaMalloc(&newData,row*col*sizeof(float));
    cudaMemcpy(newData,h_data_temp,row*col*sizeof(float),cudaMemcpyHostToDevice);


    float *distances;
    distances=(float*)malloc(sizeof(float)*rows);

    dim3 blocks=(ceil((row*col)/128));
    dim3 threads=(128);

    KNN_CUDA<<<blocks,threads>>>(deviceData,newData,rows,cols,distances);

    //code to answer that soon i will write that// 


    int prediction=majorityCOUNT(distances);

    return prediction;
    
    
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
        
        cudaMalloc(&deviceData,sizeof(float)*rows*cols);
        cudaMemcpy(deviceData,h_data_temp,sizeof(float)*rows*cols,cudaMemcpyHostToDevice);

        delete[] h_data_temp;

}


int knn::KNN::majorityCOUNT(float* distances){
    std::map<float,int> combinedDISLAB;

    std::cout<<labels.size()<<std::endl;
    for(int i=0;i<labels.size();i++){
        combinedDISLAB[distances[i]]=labels[i];
        std::cout<<distances[i]<<" "<<labels[i]<<std::endl;
    }

    auto itr=combinedDISLAB.begin();

    int ret=itr->second; //default case when no ones get majority , too sbsey kum distance walla return krunga
    float dist=itr->first;
    int count=1;
    std::map<int,int> occurCOUNT;
    occurCOUNT[ret]=1;
    int lab=0;
    itr++;

    while(itr!=combinedDISLAB.end() && lab<k_NEIG){
        if(occurCOUNT.find(itr->second)==occurCOUNT.end()){
            occurCOUNT[itr->second]=1;
        }
        else{
            occurCOUNT[itr->second]+=1;
        }

        if(count<occurCOUNT[itr->second]){
            count=occurCOUNT[itr->second];
            ret=itr->second;
            
        }
        itr++;
        lab++;

    }
    return ret;

}

std::vector<std::vector<float>> knn::KNN::csvTOvector(const std::string path){
            std::ifstream file(path);
            std::string line;

            std::vector<std::vector<float>> ret; //keeping my legacy of naming returning variable as ret (LEETCODE se sikhey h 22)
            if(file.is_open()){
                while(getline(file,line)){
                    std::stringstream ss(line);
                    std::string value;
                    std::vector<float> row;
                    while(getline(ss,value,',')){
                        row.push_back(std::stof(value));

                    }

                    std::cout<<std::endl;
                    ret.push_back(std::vector<float>(row.begin(),row.end()-1));
                   
                    labels.push_back(row[row.size()-1]);
                }
                file.close();
            }  
            else{
                std::cerr<<"Unable to open file"<<std::endl;
                
            }
            // ret.pop_back();
            // labels.pop_back();
            // for(auto k:labels){
            //     std::cout<<k<<" ";
            // }

            return ret;
}


