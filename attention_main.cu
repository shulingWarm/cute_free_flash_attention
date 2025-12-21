
#include"rand_generator.h"
#include<cuda_runtime.h>
#include <cutlass/bfloat16.h>

using bf16 = cutlass::bfloat16_t;

using u32=unsigned int;

// flash attention的核函数
template<class T, // 数据类型
    u32 HEAD_NUM=128, // 头的数量
    u32 HEAD_DIM=128, // 头的维度
    u32 MMA_M_SIZE=16,
    u32 MMA_N_SIZE=8,
    u32 MMA_K_SIZE=16,
    u32 WARP_SIZE=32, // warp size
    u32 WARP_NUM=4
>
__global__ void flash_attention(
    T* query, // [batch, num_heads, seq_len, head_dim]
    T* key, // [batch, num_heads, seq_len, head_dim]
    T* value, // [batch, num_heads, seq_len, head_dim]
    T* output, // [batch, num_heads, seq_len, head_dim]
    u32 seq_len
) {
    // 总的线程数
    constexpr u32 THREAD_NUM = WARP_NUM * WARP_SIZE; // 参考值: 4*32=128
    // 当前线程的warp id
    u32 id_warp = threadIdx.x / WARP_SIZE;
    u32 in_warp_offset = threadIdx.x % WARP_SIZE;

    // 编译检查T的大小是2字节，目前只支持bf16和fp16
    static_assert(sizeof(T) == 2, "only bf16 and fp16 are supported");

    constexpr u32 U32_MMA_K_SIZE = MMA_K_SIZE * sizeof(T) / sizeof(u32); // 参考值: 16*2/4=8
    constexpr u32 U32_HEAD_DIM = HEAD_DIM * sizeof(T) / sizeof(u32); // 参考值: 128*2/4=64

    // 每次调用mma时的矩阵的大小
    constexpr u32 MMA_A_SIZE = MMA_M_SIZE * MMA_K_SIZE; // 参考值: 16*16=256
    constexpr u32 MMA_B_SIZE = MMA_N_SIZE * MMA_K_SIZE; // 参考值: 16*8=128
    constexpr u32 MMA_C_SIZE = MMA_M_SIZE * MMA_N_SIZE; // 参考值: 16*8=128

    // 三个矩阵中每个线程准备的数据量
    constexpr u32 MMA_A_THREAD_SIZE = MMA_A_SIZE / WARP_SIZE; // 参考值: 256/32=8
    constexpr u32 MMA_B_THREAD_SIZE = MMA_B_SIZE / WARP_SIZE; // 参考值: 128/32=4
    constexpr u32 MMA_C_THREAD_SIZE = MMA_C_SIZE / WARP_SIZE; // 参考值: 128/32=4

    // MMA在K方向的循环次数
    constexpr u32 MMA_K_LOOP_NUM = HEAD_DIM / MMA_K_SIZE; // 参考值: 128/16=8

    // 用u32的形式加载时候的加载量
    constexpr u32 U32_MMA_A_THREAD_SIZE = MMA_A_THREAD_SIZE * sizeof(u32) / sizeof(T); // 参考值: 8*2/4=4
    constexpr u32 U32_MMA_B_THREAD_SIZE = MMA_B_THREAD_SIZE * sizeof(u32) / sizeof(T); // 参考值: 4*2/4=2

    // 每个线程块需要加载的query数据量
    constexpr u32 QUERY_BLOCK_LOAD_SIZE = MMA_N_SIZE * HEAD_DIM * WARP_NUM; // 参考值: 8*128*4=4096
    // 对于同一个head，划分出来的block数量 这里指的是加载的block数量
    constexpr u32 BLOCK_NUM_IN_HEAD = HEAD_DIM*sizeof(T)/128; // 参考值: 128*2/128=2
    // 每次加载KV时的加载量
    constexpr u32 KV_BLOCK_LOAD_SIZE = MMA_M_SIZE * HEAD_DIM; // 参考值: 16*128=2048
    // 将query从global复制到shared的过程中，每个线程负责的数据量
    constexpr u32 QUERY_THREAD_COPY_SIZE = (QUERY_BLOCK_LOAD_SIZE) / THREAD_NUM; // 参考值: 4096/128=32
    // 编译检查每个warp的加载量刚好等于每一组query的数据量
    static_assert(QUERY_THREAD_COPY_SIZE*sizeof(T) == MMA_N_SIZE*BLOCK_NUM_IN_HEAD*4, "QUERY_BLOCK_LOAD_SIZE must be equal to MMA_N_SIZE*BLOCK_NUM_IN_HEAD*4");
    constexpr u32 U32_QUERY_THREAD_COPY_SIZE = QUERY_THREAD_COPY_SIZE * sizeof(u32) / sizeof(T); // 参考值: 32*2/4=16
    // 执行mma计算的时候寄存器的使用量
    constexpr u32 THREAD_REG_SIZE = MMA_A_THREAD_SIZE + MMA_B_THREAD_SIZE + MMA_C_THREAD_SIZE; // 参考值: 8+4+4=16

    // 准备三个矩阵的寄存器，用于参与mma计算
    T all_reg_ptr[THREAD_REG_SIZE];
    T* mma_a_reg = all_reg_ptr;
    T* mma_b_reg = mma_a_reg + MMA_A_THREAD_SIZE;
    T* mma_c_reg = mma_b_reg + MMA_B_THREAD_SIZE;
    u32* u32_mma_a_reg = (u32*)mma_a_reg;
    u32* u32_mma_b_reg = (u32*)mma_b_reg;
    u32* u32_mma_c_reg = (u32*)mma_c_reg;

    // 一次加载两个数据的情况下，query需要加载的次数
    constexpr u32 QUERY_THREAD_COPY_U32 = QUERY_THREAD_COPY_SIZE * sizeof(T) / sizeof(u32); // 参考值: 16*2/4=8

    // KV的seq_len长度的循环次数
    u32 seq_len_loop_num = seq_len / MMA_A_SIZE;
    // head_dim维度的循环次数
    constexpr u32 HEAD_DIM_LOOP_NUM = HEAD_DIM / MMA_K_SIZE; // 参考值: 128/16=8

    // 用于记录query的shared memory
    // 关于KV_LOAD_SIZE*2 表示分别加载KV
    __shared__ T all_shared[QUERY_BLOCK_LOAD_SIZE + KV_BLOCK_LOAD_SIZE];
    // query版本的32头指针
    u32* query_shared_u32_ptr = (u32*)all_shared;

    // 将query从全局内存复制到共享内存
    {
        // 当前线程访问的query数据的头指针
        T* thread_query_head = query + blockIdx.z * seq_len * HEAD_NUM * HEAD_DIM + 
            blockIdx.y * seq_len * HEAD_DIM + 
            (blockIdx.x * WARP_NUM + id_warp) * HEAD_DIM * MMA_N_SIZE;
        u32* u32_thread_query_head = (u32*)thread_query_head;
        // 当前warp访问的共享内存的起始地址
        T* query_shared_head = all_shared + id_warp * MMA_N_SIZE * HEAD_DIM;
        u32* u32_query_shared_head = (u32*)query_shared_head;

        // 用于复制全局内存的寄存器
        u32 query_copy_reg[U32_QUERY_THREAD_COPY_SIZE];

        // 每个线程按顺序读取数据
        for(u32 id_seq=0;id_seq<MMA_N_SIZE; ++id_seq) {
            // 计算本线程应该在当前block中处理的位置
            u32 offset_in_block = (id_seq&3) ^ (in_warp_offset >> 3);
            // 对于每个序列，遍历它的每个block
            #pragma unroll
            for(u32 id_block=0;id_block<BLOCK_NUM_IN_HEAD; ++id_block) {
                asm volatile(
                    "ld.global.cg.b32 %0, [%1];\n"
                    : "=r"(query_copy_reg[id_seq*BLOCK_NUM_IN_HEAD + id_block])
                    : "l"(u32_thread_query_head + 
                        id_seq*U32_HEAD_DIM + 
                        id_block*WARP_SIZE + 
                        offset_in_block*U32_MMA_K_SIZE + 
                        (in_warp_offset%U32_MMA_K_SIZE))
                )
            }
        }

        // 将query从寄存器复制到共享内存
        for(u32 id_mma_loop=0;id_mma_loop<MMA_K_LOOP_NUM; ++id_mma_loop) {
            // 遍历当前mma切片的每个block
            for(u32 id_block=0;id_block<BLOCK_NUM_IN_HEAD; ++id_block) {
                // 计算当前线程持有的数据在当前切片中属于哪一行
                u32 in_block_row = ((in_warp_offset>>3) ^ (id_mma_loop&3))&3;
                // 将数据写入到共享内存
                u32_query_shared_head[(id_block_row + 4*id_block)*U32_HEAD_DIM + 
                    id_mma_loop*8 + in_warp_offset&7] = 
                    query_copy_reg[(in_block_row + 4*id_block)*2 + id_mma_loop/4];
            }
        }
    }


    // blockIdx.x 是query_pick方向的id
    // blockIdx.y 是head_id
    // blockIdx.z 是batch_id

    // 当前的线程块要加载的query tensor的头指针
    T* query_block_ptr = query + blockIdx.z * seq_len * HEAD_NUM * HEAD_DIM + 
        blockIdx.y * seq_len * HEAD_DIM + 
        blockIdx.x * HEAD_DIM * SEQ_PICK_NUM;
    // 当前block的kv 头指针
    T* key_block_ptr = key + blockIdx.z * seq_len * HEAD_NUM * HEAD_DIM + 
        blockIdx.y * seq_len * HEAD_DIM;
    T* value_block_ptr = value + blockIdx.z * seq_len * HEAD_NUM * HEAD_DIM + 
        blockIdx.y * seq_len * HEAD_DIM;

    // 依次遍历每个需要被复制到寄存器的数据
    #pragma unroll
    for(u32 id_query_copy=0;id_query_copy<QUERY_THREAD_COPY_U32; ++id_query_copy) {
        asm volatile(
            "ld.global.cg.b32 %0, [%1];\n"
            : "=r"(u32_mma_a_reg[id_query_copy])
            : "l"(query_block_ptr + id_query_copy * THREAD_NUM + threadIdx.x)
        );
    }

    // 将query从寄存器复制到共享内存中
    {
        // 当前warp的线程复制到目标共享内存的起始地址
        u32* query_copy_target_for_warp = query_shared_u32_ptr + 
            id_warp * QUERY_THREAD_COPY_U32*WARP_SIZE + in_warp_offset;

        #pragma unroll
        for(u32 id_query_copy=0; id_query_copy<QUERY_THREAD_COPY_U32; ++id_query_copy) {
            query_copy_target_for_warp[id_query_copy*WARP_SIZE] = 
                u32_mma_a_reg[id_query_copy];
        }
    }

    // 复制K数据
    {
        // 当前warp的头指针
        T* key_head = key_block_ptr + id_warp * WARP_SIZE*KV_LOAD_THREAD_SIZE + in_warp_offset*sizeof(u32)/sizeof(T);
        u32* u32_key_head = (u32*)key_head;
        // 依次将每个数据从全局内存复制到寄存器里面
        #pragma unroll
        for(u32 id_load=0;id_load<U32_KV_LOAD_THREAD_SIZE; ++id_load) {
            asm volatile(
                "ld.global.cg.b32 %0, [%1];\n"
                : "=r"(u32_kv_load_reg[id_load])
                : "l"(u32_key_head + id_load * WARP_SIZE)
            );
        }

        // 将寄存器中的数据复制到共享内存
        T* warp_shared_head = kv_shared_1 + id_warp * KV_LOAD_THREAD_SIZE*WARP_SIZE + KV_LOAD_THREAD_SIZE*in_warp_offset;
        u32* u32_warp_shared_head = (u32*)warp_shared_head;
        #pragma unroll
        for(u32 id_store=0; id_store<U32_KV_LOAD_THREAD_SIZE; ++id_store) {
            u32_warp_shared_head[id_store*WARP_SIZE] = u32_kv_load_reg[id_store];
        }
    }
    

    // 执行计算的主循环体
    // 序列方向的循环层
    for(u32 id_seq_step=0; id_seq_step<seq_len_loop_num; ++id_seq_step) {
        // 调用同步
        __syncthreads();

        // 遍历K方向的每一次计算
        for(u32 id_compute=0;id_compute<HEAD_DIM_LOOP_NUM; ++id_compute) {
            // 从共享内存里面读取query
            #pragma unroll
            for(u32 id_query_load=0; id_query_load<U32_MMA_A_THREAD_SIZE; ++id_query_load) {
                // 当前实际要加载的id
                // 对于线程0~15，id是0,1,2,3
                // 对于线程16~31, id是2,3,0,1
                u32 id_query_load_real = (id_query_load + ((id_query_load&0x10)>>3))&0x3;
                // 在共享内存里访问位置的偏移量:
                // ((id%2)*8+tid/4)*8 + (id/2)*4 + tid%4
                // 用位运算化简之后的表达式：((id & 1) << 6) | ((tid & 0x1C) << 1) | ((id & 2) << 1) | (tid & 3)
                asm volatile(
                    "ld.shared.cg.b32 %0, [%1];\n"
                    : "=r"(u32_mma_a_reg[id_query_load_real])
                    : "l"(query_shared_u32_ptr + (((id_query_load_real & 1) << 6) | 
                        ((in_warp_offset & 0x1C) << 1) | 
                        ((id_query_load_real & 2) << 1) | 
                        (in_warp_offset & 3)))
                );
            }

            // 从共享内存里读取K
            #pragma unroll
            for(u32 id_load=0;id_load<U32_MMA_B_THREAD_SIZE; ++id_load) {
                
            }
        }
    }
}

int main() {
    // 初始化attention的四个维度, batch, seq_len, head_num, head_dim
    constexpr u32 BATCH_NUM = 1;
    constexpr u32 SEQ_LEN = 4096;
    constexpr u32 HEAD_NUM = 32;
    constexpr u32 HEAD_DIM = 128;
    constexpr u32 TOTAL_SIZE = BATCH_NUM * SEQ_LEN * HEAD_NUM * HEAD_DIM;
    constexpr u32 THREAD_PER_BLOCK = 128;

    using MainType = bf16;

    // 初始化qkvo 四个tensor，大小都相同 
    MainType* query = (MainType*)malloc(TOTAL_SIZE * sizeof(MainType));
    MainType* key = (MainType*)malloc(TOTAL_SIZE * sizeof(MainType));
    MainType* value = (MainType*)malloc(TOTAL_SIZE * sizeof(MainType));
    MainType* output = (MainType*)malloc(TOTAL_SIZE * sizeof(MainType));

    // 随机初始化qkv的值
    UniformRandomGenerator rand_gen;
    init_matrix_with_length<MainType>(query, TOTAL_SIZE, rand_gen);
    init_matrix_with_length<MainType>(key, TOTAL_SIZE, rand_gen);
    init_matrix_with_length<MainType>(value, TOTAL_SIZE, rand_gen);

    // 调用flash attention核函数，每个线程块128个线程
    dim3 grid_dim(SEQ_LEN / 16, HEAD_NUM, BATCH_NUM);
    dim3 block_dim(THREAD_PER_BLOCK, 1, 1);
    flash_attention<MainType, // 数据类型
        HEAD_NUM, // 头的数量
        HEAD_DIM, // 头的维度
        16, // MMA_M_SIZE
        16, // MMA_N_SIZE
        16  // MMA_K_SIZE
    ><<<grid_dim, block_dim>>>(
        query,
        key,
        value,
        output,
        SEQ_LEN
    );
}