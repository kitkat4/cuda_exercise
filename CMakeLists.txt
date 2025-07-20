cmake_minimum_required(VERSION 3.22)

project(cuda_exercise LANGUAGES C CXX CUDA)

# 複数のアーキテクチャをセミコロンで結合して指定できる。
# 指定したバージョンがnvccの--generate-codeオプションに渡され、
# 指定したそれぞれのアーキテクチャに最適化されたコード
# （すなわち、新しいアーキテクチャには新しい機能を使ったコード）が生成・結合される。
# 実行時には実行環境に合わせたコードが選択され、実行される。
set(CMAKE_CUDA_ARCHITECTURES 75;89)

add_executable(01_hello_world src/01_hello_world.cu)

add_executable(02_add_float_array_cpu src/02_add_float_array_cpu.c)

add_executable(03_add_float_array_gpu src/03_add_float_array_gpu.cu)

add_executable(04_add_matrix src/04_add_matrix.cu)


