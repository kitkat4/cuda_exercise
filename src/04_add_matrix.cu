#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

typedef enum{
    GPU = 0,
    CPU
}PROC_TYPE;

#define CUDA_CHECK(call)                                                \
    {                                                                   \
        const cudaError_t error = call;                                 \
        if(error != cudaSuccess){                                       \
            printf("Error: %s:%d    ", __FILE__, __LINE__);             \
            printf("code: %d, reason: %s\n", error,                     \
                   cudaGetErrorString(error));                          \
            exit(1);                                                    \
        }                                                               \
    }


void initMat(const std::shared_ptr<float>& mat, const int mat_size){

    if(! mat){
        throw std::runtime_error("Invalid pointer");
    }

    std::random_device seed_gen;
    std::mt19937 rand_engine(seed_gen());

    for(int i = 0; i < mat_size; i++){
        for(int j = 0; j < mat_size; j++){
            mat.get()[i * mat_size + j] = rand_engine() / (float)(std::mt19937::max());
        }
    }

    return;
}

void printMat(const std::shared_ptr<float>& mat, const int mat_size){

    for(int i = 0; i < mat_size; i++){
        for(int j = 0; j < mat_size; j++){
            std::cout << mat.get()[i * mat_size + j] << ", ";
        }
        std::cout << std::endl;
    }
}

void printAdditionResults(const std::shared_ptr<float>& mat1,
                          const std::shared_ptr<float>& mat2,
                          const std::shared_ptr<float>& mat3,
                          const int mat_size){

    for(int i = 0; i < mat_size; i++){
        for(int j = 0; j < mat_size; j++){
            std::cout << "[" << i << ", " << j << "] " << mat3.get()[i * mat_size + j] << " = " << mat1.get()[i * mat_size + j] << " + " << mat2.get()[i * mat_size + j] << " (error: " << mat1.get()[i * mat_size + j] + mat2.get()[i * mat_size + j] - mat3.get()[i * mat_size + j] << ")" << std::endl;
        }
    }
}

__global__ void addMat(float* mat1, float* mat2, float* mat3, const int mat_size){

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx_x < 0 || mat_size <= idx_x ||
       idx_y < 0 || mat_size <= idx_y){
        return;
    }
    mat3[idx_y * mat_size + idx_x] = mat1[idx_y * mat_size + idx_x] + mat2[idx_y * mat_size + idx_x];
}

int main(int argc, char** argv){

    if(argc != 5){
        std::cerr << "Usage: " << argv[0] << " <0/1 (0: gpu, 1: cpu)> <matrix size (num of cols and rows)> <block size x> <block size y>" << std::endl;
        return 1;
    }

    const PROC_TYPE proc_type = (PROC_TYPE)(std::stoi(argv[1]));
    const int mat_size = std::stoi(argv[2]);
    const int block_x = std::stoi(argv[3]);
    const int block_y = std::stoi(argv[4]);
    const int grid_x = (mat_size + block_x - 1) / block_x;
    const int grid_y = (mat_size + block_y - 1) / block_y;

    std::shared_ptr<float> mat1(new float[mat_size * mat_size]);
    std::shared_ptr<float> mat2(new float[mat_size * mat_size]);
    std::shared_ptr<float> mat3(new float[mat_size * mat_size]);

    initMat(mat1, mat_size);
    initMat(mat2, mat_size);

    if(proc_type == GPU){

        const int device = 0;
        cudaDeviceProp device_prop;
        CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device));
        std::cout << "Name: " << device_prop.name << std::endl;
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        std::cout << "Total memory [GB]: " << total_mem / 1e9 << std::endl;
        std::cout << "Free  memory [GB]: " << free_mem / 1e9 << " (" << 100.0 * free_mem / (double)total_mem << " %)" << std::endl;
        CUDA_CHECK(cudaSetDevice(device));
        
        float* gpu_mat1 = nullptr;
        float* gpu_mat2 = nullptr;
        float* gpu_mat3 = nullptr;
        CUDA_CHECK(cudaMalloc(&gpu_mat1, mat_size * mat_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_mat2, mat_size * mat_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_mat3, mat_size * mat_size * sizeof(float)));        

        CUDA_CHECK(cudaMemcpy(gpu_mat1, mat1.get(),
                              mat_size * mat_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_mat2, mat2.get(),
                              mat_size * mat_size * sizeof(float), cudaMemcpyHostToDevice));

        dim3 grid_size(grid_x, grid_y);
        dim3 block_size(block_x, block_y);        

        addMat<<<grid_size, block_size>>>(gpu_mat1, gpu_mat2, gpu_mat3, mat_size);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(mat3.get(), gpu_mat3,
                              mat_size * mat_size * sizeof(float), cudaMemcpyDeviceToHost));
        
    }else if(proc_type == CPU){
        for(int i = 0; i < mat_size; i++){
            for(int j = 0; j < mat_size; j++){
                mat3.get()[i * mat_size + j] =
                    mat1.get()[i * mat_size + j] + mat2.get()[i * mat_size + j];
            }
        }
    }else{
        throw std::runtime_error("Invalid option");
    }

    printAdditionResults(mat1, mat2, mat3, mat_size);

    return 0;
}

