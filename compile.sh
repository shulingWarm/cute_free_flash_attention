

nvcc -I/mnt/data/code/flash-attention/csrc/cutlass/include -O3 \
    -arch=sm_89 attention_main.cu -o attention_main \
    --ptxas-options=-v

