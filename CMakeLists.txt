cmake_minimum_required(VERSION 3.10)

project(MyProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

enable_language(CUDA)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCE_FILES
    main.cpp
    src/knn_cuda.cpp
)

add_executable(MyProject ${SOURCE_FILES})

target_include_directories(MyProject PRIVATE ${CMAKE_SOURCE_DIR}/cuda_include)
