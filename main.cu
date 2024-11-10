#include "inc//knn_cuda.hpp"


int main(){
    std::string path="train.txt";
    knn::KNN model(path,3);

    model.fit();


    std::cout<<model.predict("test.txt")<<"done"<<std::endl;

    return 0;



}