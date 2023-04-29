cmake_minimum_required(VERSION 3.17)

project(cs267-final LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HDF5 REQUIRED)

# Use the project root to find includes
include_directories(${PROJECT_SOURCE_DIR})

add_executable(main)
target_sources(main PRIVATE src/main.cu)
target_link_libraries(main PRIVATE hdf5 hdf5_cpp)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "80")
