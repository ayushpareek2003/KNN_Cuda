#include "..\inc\knn_cuda.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <fstream>
#include <sstream>

__global__ void KNN_CUDA(float *deviceData, float *testData, int rows, int cols, float *distances) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows) {
        float dist = 0.0;
        for (int j = 0; j < cols; ++j) {
            float diff = testData[j] - deviceData[tid * cols + j];
            dist += diff * diff;
        }
        distances[tid] = sqrt(dist);
    }
}

knn::KNN::KNN(const std::string pathToCSV, int k_NEIG, bool is_cudaTrue, int distanceType)
    : is_cudaTrue(is_cudaTrue), distanceType(distanceType), pathToCSV(pathToCSV), k_NEIG(k_NEIG) {
    hostData = csvTOvector(pathToCSV);
}

int knn::KNN::predict(const std::string testData) {
    std::vector<std::vector<float>> readyTestData = csvTOvector(testData);
    if (readyTestData.empty()) return -1;

    int row = readyTestData.size();
    int col = readyTestData[0].size();
    std::vector<float> h_testData(readyTestData[0].begin(), readyTestData[0].end());
    
    float *d_testData, *distances,*d_trainData;
    cudaMalloc(&d_testData, row *col * sizeof(float));
    cudaMalloc(&d_trainData,rows*cols* sizeof(float));
    cudaMemcpy(d_testData, h_testData.data(), col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trainData, h_testData.data(), col * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&distances, rows * sizeof(float));
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;

    KNN_CUDA<<<blocks, threads>>>(d_trainData, d_testData, rows, cols, distances);
    cudaDeviceSynchronize();

    std::vector<float> h_distances(rows);
    cudaMemcpy(h_distances.data(), distances, rows * sizeof(float), cudaMemcpyDeviceToHost);

    int prediction = majorityCOUNT(h_distances.data());

    cudaFree(d_testData);
    cudaFree(distances);

    return prediction;
}

void knn::KNN::fit() {
    if (is_cudaTrue) {
        transferDataToDevice();
    } else {
        std::cout << "CUDA is disabled. Running on CPU." << std::endl;
    }
}

void knn::KNN::transferDataToDevice() {
    rows = hostData.size();
    cols = rows > 0 ? hostData[0].size() : 0;

    std::vector<float> h_data_temp(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_data_temp[i * cols + j] = hostData[i][j];
        }
    }

    cudaMalloc(&deviceData, rows * cols * sizeof(float));
    cudaMemcpy(deviceData, h_data_temp.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

int knn::KNN::majorityCOUNT(float* distances) {
    std::map<float, int> combinedDISLAB;
    for (int i = 0; i < labels.size(); ++i) {
        combinedDISLAB[distances[i]] = labels[i];
    }

    int ret = combinedDISLAB.begin()->second;
    int count = 1;
    std::map<int, int> occurCOUNT;
    occurCOUNT[ret] = 1;

    auto itr = combinedDISLAB.begin();
    for (int lab = 0; itr != combinedDISLAB.end() && lab < k_NEIG; ++itr, ++lab) {
        occurCOUNT[itr->second]++;
        if (occurCOUNT[itr->second] > count) {
            count = occurCOUNT[itr->second];
            ret = itr->second;
        }
    }
    return ret;
}

std::vector<std::vector<float>> knn::KNN::csvTOvector(const std::string path) {
    std::ifstream file(path);
    std::string line;
    std::vector<std::vector<float>> ret;

    if (file.is_open()) {
        while (getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            std::vector<float> row;
            while (getline(ss, value, ',')) {
                row.push_back(std::stof(value));
            }
            ret.push_back(std::vector<float>(row.begin(), row.end() - 1));
            labels.push_back(row.back());
        }
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }

    return ret;
}
