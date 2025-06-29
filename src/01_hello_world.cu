#include <stdio.h>
#include <iostream>

#include "../include/my_utils_kk4.hpp"

__global__ void helloFromGPU(){
    printf("Hello world from GPU\n");
    // std::cout << "Hello world from GPU" << std::endl;  これは動作しない
}

int main(int argc, char** argv){

    my_utils_kk4::StopWatch sw;
    
    // printf("Hello world from CPU\n");
    std::cout << "Hello world from CPU" << std::endl;

    sw.start();
    
    helloFromGPU<<<1, 10>>>();

    cudaDeviceReset();

    std::cout << sw.stop() * 1000.0 << " ms" << std::endl;

    return 0;
}



