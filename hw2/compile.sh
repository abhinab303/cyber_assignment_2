# g++ hop_ann.cc -o hop_ann -std=c++11
nvcc hop_ann_cuda.cu -o hop_ann -Xcompiler -fopenmp
# nvcc hop_ann_cuda.cu -o hop_ann -Xcompiler -fopenmp -O3
# nvcc hop_ann_cuda.cu -o hop_ann -Xcompiler -fopenmp -O3 -Xptxas -O3 -arch=native