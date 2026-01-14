
#include"rand_generator.h"
#include<cuda_runtime.h>
#include <cutlass/bfloat16.h>

using bf16 = cutlass::bfloat16_t;

using u32=unsigned int;

#define DEBUG_FLAG

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
#ifdef DEBUG_FLAG
    ,
    u32* debug_tensor // 用来存储中间结果
#endif
) {
    // 总的线程数
    constexpr u32 THREAD_NUM = WARP_NUM * WARP_SIZE; // 参考值: 4*32=128
    // 当前线程的warp id
    u32 id_warp = threadIdx.x / WARP_SIZE;
    u32 in_warp_offset = threadIdx.x % WARP_SIZE;

    // MMA指令使用时候的行列数
    u32 MMA_ROW_NUM = 2;
    u32 MMA_COL_NUM = 2;

    // 编译检查T的大小是2字节，目前只支持bf16和fp16
    static_assert(sizeof(T) == 2, "only bf16 and fp16 are supported");

    // mma计算过程中的layout k方向的u32粒度
    constexpr u32 K_LAYOUT_UNIT = 4;
    // 当前warp对应的十字块的行号和列号
    u32 four_block_row_id = id_warp % 2;
    u32 four_block_col_id = id_warp / 2;

    constexpr u32 U32_MMA_K_SIZE = MMA_K_SIZE * sizeof(T) / sizeof(u32); // 参考值: 16*2/4=8
    constexpr u32 U32_HEAD_DIM = HEAD_DIM * sizeof(T) / sizeof(u32); // 参考值: 128*2/4=64
    // 每个warp一次能读取几个K block
    constexpr u32 K_BLOCK_NUM_PER_WARP = (WARP_SIZE / K_LAYOUT_UNIT); // 参考值: 32/4=8
    constexpr u32 K_LAYOUT_LOOP_NUM = U32_HEAD_DIM / K_LAYOUT_UNIT; // 参考值: 64/4=16

    // 每个十字块的行数和列数
    constexpr u32 FOUR_BLOCK_ROW_SIZE_KEY = MMA_M_SIZE / 2;
    constexpr u32 FOUR_BLOCK_COL_SIZE = U32_HEAD_DIM / 2;
    // 每个十字块里面的unit块数
    constexpr u32 UNIT_NUM_IN_FOUR_BLOCK = FOUR_BLOCK_COL_SIZE / K_LAYOUT_UNIT; // 参考值: 32/4=8
    constexpr u32 UNIT_NUM_IN_HEAD = U32_HEAD_DIM / K_LAYOUT_UNIT; // 参考值: 64/4=16
    // 用static assert确保 UNIT_NUM_IN_FOUR_BLOCK 和 FOUR_BLOCK_ROW_SIZE_KEY 相同
    static_assert(FOUR_BLOCK_ROW_SIZE_KEY == UNIT_NUM_IN_FOUR_BLOCK, "FOUR_BLOCK_ROW_SIZE_KEY must be equal to UNIT_NUM_IN_FOUR_BLOCK");

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
    // mma的输出矩阵的寄存器
    constexpr u32 THREAD_OUTPUT_REG_SIZE = MMA_N_SIZE*HEAD_DIM/WARP_SIZE; // 参考值: 8*128/32=32
    constexpr u32 THREAD_OUTPUT_STEP_REG = THREAD_OUTPUT_REG_SIZE/MMA_K_LOOP_NUM; // 参考值: 32/8=4
    // 编译时检查 MMA_K_LOOP_NUM*(MMA_M_SIZE*MMA_N_SIZE)/WARP_SIZE 是否等于 THREAD_OUTPUT_REG_SIZE
    static_assert(MMA_K_LOOP_NUM*(MMA_M_SIZE*MMA_N_SIZE)/WARP_SIZE == THREAD_OUTPUT_REG_SIZE, 
        "THREAD_OUTPUT_REG_SIZE must be equal to MMA_K_LOOP_NUM*(MMA_M_SIZE*MMA_N_SIZE)/WARP_SIZE");

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

    // 每个线程里面mma输出矩阵的寄存器
    float mma_output_reg[THREAD_OUTPUT_REG_SIZE] = {0};

    // KV加载的周期数
#ifdef DEBUG_FLAG
    u32 KV_LOAD_LOOP_NUM = 1;
#else
    u32 KV_LOAD_LOOP_NUM = seq_len / MMA_M_SIZE;
#endif
    // 每个warp需要读取的行数
    u32 KV_LOAD_ROW_NUM_PER_WARP = MMA_M_SIZE / WARP_NUM;
    // 每个warp在每一行中需要读取的块数
    u32 KV_LOAD_ROW_BLOCK_NUM = HEAD_DIM*sizeof(T)/sizeof(u32)/WARP_SIZE; // 参考值: 128*2/4/32=2;
    // 加载KV的时候，每个线程需要加载的数据量
    constexpr u32 U32_KV_LOAD_PER_THREAD = KV_BLOCK_LOAD_SIZE/THREAD_NUM*sizeof(T)/sizeof(u32); // 参考值: 2048/128*2/4=8

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
    // key版本的32位头指针
    u32* key_shared_u32_ptr = (u32*)(all_shared + QUERY_BLOCK_LOAD_SIZE);

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
            u32 offset_in_block = id_seq ^ (in_warp_offset / K_LAYOUT_UNIT);
            // 对于每个序列，遍历它的每个block
            #pragma unroll
            for(u32 id_block=0;id_block<BLOCK_NUM_IN_HEAD; ++id_block) {
                asm volatile(
                    "ld.global.cg.b32 %0, [%1];\n"
                    : "=r"(query_copy_reg[id_seq*BLOCK_NUM_IN_HEAD + id_block])
                    : "l"(u32_thread_query_head + 
                        id_seq*U32_HEAD_DIM + 
                        id_block*WARP_SIZE + 
                        offset_in_block*K_LAYOUT_UNIT + 
                        (in_warp_offset%K_LAYOUT_UNIT))
                );
            }
        }

        // 将query从寄存器复制到共享内存
        for(u32 id_mma_loop=0;id_mma_loop<UNIT_NUM_IN_HEAD; ++id_mma_loop) {
            // 计算当前线程持有的数据在当前切片中属于哪一行
            u32 in_block_row = ((in_warp_offset/K_LAYOUT_UNIT) ^ (id_mma_loop%UNIT_NUM_IN_FOUR_BLOCK));
            // 将数据写入到共享内存
            u32_query_shared_head[(in_warp_offset%K_LAYOUT_UNIT) + 
                (in_block_row + id_mma_loop*MMA_N_SIZE)*K_LAYOUT_UNIT] = 
                query_copy_reg[in_block_row*BLOCK_NUM_IN_HEAD + id_mma_loop/UNIT_NUM_IN_FOUR_BLOCK];
        }

#ifdef DEBUG_FLAG
        // 调用线程同步
        __syncthreads();
        // 将某个block里面的寄存器数据写入debug_tensor
        if(blockIdx.x==12 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x==0) {
            printf("begin write key debug tensor\n");
            u32* u32_all_shared = (u32*)all_shared;
            // 把warp 0 里面的寄存器数据写入到debug_tensor
            // 一个warp在共享内存里面负责的数据量是: 64*8 = 512
            for(u32 id_data=0;id_data<2048;++id_data) {
                debug_tensor[id_data] = u32_all_shared[id_data];
            }
        }
#endif
    }

    // 计算过程的主循环
    for(u32 id_loop=0;id_loop<KV_LOAD_LOOP_NUM; ++id_loop) {
        // 将key从全局内存加载到共享内存
        {
            // 当前轮次访问key数据的头指针
            T* key_head_ptr = key + blockIdx.z * seq_len * HEAD_NUM * HEAD_DIM + 
                (id_loop*MMA_M_SIZE) * HEAD_DIM;
            u32* u32_key_head_ptr = ((u32*)key_head_ptr) + four_block_col_id*FOUR_BLOCK_COL_SIZE +
                four_block_row_id*FOUR_BLOCK_ROW_SIZE_KEY*U32_HEAD_DIM;

            // 共享内存的头指针
            u32* warp_shared_key_head = key_shared_u32_ptr + 
                four_block_col_id*MMA_M_SIZE*FOUR_BLOCK_COL_SIZE +
                four_block_row_id*FOUR_BLOCK_ROW_SIZE_KEY*K_LAYOUT_UNIT;

            // 用于复制全局内存的寄存器
            u32 key_copy_reg[FOUR_BLOCK_ROW_SIZE_KEY];

            // 将key数据从全局内存复制到寄存器
            // 遍历每个warp要读取的每一行
            for(u32 id_row=0;id_row<FOUR_BLOCK_ROW_SIZE_KEY; ++id_row) {
                // 计算当前线程在当前行所在的偏移量位置
                u32 offset_in_block = (id_row^(in_warp_offset/K_LAYOUT_UNIT))%UNIT_NUM_IN_FOUR_BLOCK;
                asm volatile(
                    "ld.global.cg.b32 %0, [%1];\n"
                    : "=r"(key_copy_reg[id_row])
                    : "l"(u32_key_head_ptr + 
                        id_row*U32_HEAD_DIM + 
                        offset_in_block*K_LAYOUT_UNIT + in_warp_offset%K_LAYOUT_UNIT
                    )
                );
            }

            // 将key数据从寄存器写入到共享内存
            for(u32 id_mma_loop=0;id_mma_loop<UNIT_NUM_IN_FOUR_BLOCK; ++id_mma_loop) {
                // 计算当前线程在当前mma切片中属于哪一行
                u32 in_block_row = ((in_warp_offset/K_LAYOUT_UNIT)^(id_mma_loop))%FOUR_BLOCK_ROW_SIZE_KEY;
                // 将寄存器的数据写入到共享内存中
                warp_shared_key_head[(id_mma_loop*MMA_M_SIZE + 
                    in_block_row)*K_LAYOUT_UNIT + in_warp_offset%K_LAYOUT_UNIT] =
                    key_copy_reg[in_block_row];
            }
        }

        // 调用同步确保query key加载完成
        __syncthreads();

        // mma过程中A矩阵的寄存器
        u32 mma_a_reg[U32_MMA_A_THREAD_SIZE];
        // mma过程中B矩阵的寄存器
        u32 mma_b_reg[U32_MMA_B_THREAD_SIZE];

        // Q*K^T计算过程的主循环
        for(u32 id_qkt_loop=0;id_qkt_loop<MMA_K_LOOP_NUM;++id_qkt_loop) {
            // 当前循环层中key矩阵的头指针
            u32* key_shared_head = (key_shared_u32_ptr) + 
                id_qkt_loop*U32_MMA_A_THREAD_SIZE*WARP_SIZE + in_warp_offset;
            // 当前循环层中query矩阵的头指针
            u32* query_shared_head = (query_shared_u32_ptr) + id_warp*MMA_N_SIZE*U32_HEAD_DIM +
                id_qkt_loop*U32_MMA_B_THREAD_SIZE*WARP_SIZE + in_warp_offset;
            // 准备当前循环层的输出寄存器
            u32* mma_reg_curr_loop = mma_output_reg + id_qkt_loop*THREAD_OUTPUT_STEP_REG;
            // 遍历线程要读取的每个数据
            for(u32 id_mma_read=0;id_mma_read<U32_MMA_A_THREAD_SIZE;++id_mma_read) {
                mma_a_reg[id_mma_read] = key_shared_head[id_mma_read*WARP_SIZE];
            }
            // 遍历每个线程要读取的query数据
            for(u32 id_mma_read=0;id_mma_read<U32_MMA_B_THREAD_SIZE;++id_mma_read) {
                mma_b_reg[id_mma_read] = query_shared_head[id_mma_read*WARP_SIZE];
            }

            // 执行mma计算
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10, %11, %12, %13};\n"
                : "=f"(d_fragment[0]), "=f"(d_fragment[1]), "=f"(d_fragment[2]), "=f"(d_fragment[3])
                :  "r"(a_fragment[0]),  "r"(a_fragment[1]),  "r"(a_fragment[2]),  "r"(a_fragment[3]),
                    "r"(b_fragment[0]),  "r"(b_fragment[1]),
                    "f"(c_fragment[0]),  "f"(c_fragment[1]),  "f"(c_fragment[2]),  "f"(c_fragment[3]));
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

    // 使用debug矩阵的情况下，将矩阵改成使用顺序赋值
    constexpr bool USE_DEBUG_MAT = 
#ifdef DEBUG_FLAG
        true;
#else
        false;
#endif

    using MainType = unsigned short int;

    // 初始化qkvo 四个tensor，大小都相同 
    MainType* query = (MainType*)malloc(TOTAL_SIZE * sizeof(MainType));
    MainType* key = (MainType*)malloc(TOTAL_SIZE * sizeof(MainType));
    MainType* value = (MainType*)malloc(TOTAL_SIZE * sizeof(MainType));
    MainType* output = (MainType*)malloc(TOTAL_SIZE * sizeof(MainType));

    // 随机初始化qkv的值
    UniformRandomGenerator rand_gen;
    if (USE_DEBUG_MAT) {
        init_matrix_with_order<MainType>(query, TOTAL_SIZE, 4096);
        init_matrix_with_order<MainType>(key, TOTAL_SIZE, 4096);
        init_matrix_with_order<MainType>(value, TOTAL_SIZE, 4096);
    } else {
        init_matrix_with_length<MainType>(query, TOTAL_SIZE, rand_gen);
        init_matrix_with_length<MainType>(key, TOTAL_SIZE, rand_gen);
        init_matrix_with_length<MainType>(value, TOTAL_SIZE, rand_gen);
    }

    // 初始化用于debug的矩阵
#ifdef DEBUG_FLAG
    // 初始化cuda版本的矩阵 用cudaMalloc
    u32* debug_tensor;
    constexpr u32 DEBUG_TENSOR_SIZE = 32 * 64;
    cudaMalloc((void**)&debug_tensor, DEBUG_TENSOR_SIZE * sizeof(u32));
#endif

    // 把qkvo转移到gpu上
    MainType* query_gpu;
    cudaMalloc((void**)&query_gpu, TOTAL_SIZE * sizeof(MainType));
    cudaMemcpy(query_gpu, query, TOTAL_SIZE * sizeof(MainType), cudaMemcpyHostToDevice);
    MainType* key_gpu;
    cudaMalloc((void**)&key_gpu, TOTAL_SIZE * sizeof(MainType));
    cudaMemcpy(key_gpu, key, TOTAL_SIZE * sizeof(MainType), cudaMemcpyHostToDevice);
    MainType* value_gpu;
    cudaMalloc((void**)&value_gpu, TOTAL_SIZE * sizeof(MainType));
    cudaMemcpy(value_gpu, value, TOTAL_SIZE * sizeof(MainType), cudaMemcpyHostToDevice);
    MainType* output_gpu;
    cudaMalloc((void**)&output_gpu, TOTAL_SIZE * sizeof(MainType));
    cudaMemcpy(output_gpu, output, TOTAL_SIZE * sizeof(MainType), cudaMemcpyHostToDevice);


    // 调用flash attention核函数，每个线程块128个线程
    dim3 grid_dim(SEQ_LEN / (8*4), HEAD_NUM, BATCH_NUM);
    dim3 block_dim(THREAD_PER_BLOCK, 1, 1);

    std::cout<<"grid size: "<<grid_dim.x<<" "<<grid_dim.y<<" "<<grid_dim.z<<std::endl;
    std::cout<<"block size: "<<block_dim.x<<" "<<block_dim.y<<" "<<block_dim.z<<std::endl;
    flash_attention<MainType, // 数据类型
        HEAD_NUM, // 头的数量
        HEAD_DIM, // 头的维度
        16, // MMA_M_SIZE
        8, // MMA_N_SIZE
        16  // MMA_K_SIZE
    ><<<grid_dim, block_dim>>>(
        query_gpu,
        key_gpu,
        value_gpu,
        output_gpu,
        SEQ_LEN
#ifdef DEBUG_FLAG
        , debug_tensor
#endif
    );

    // cuda的设备同步
    cudaDeviceSynchronize();

#ifdef DEBUG_FLAG
    // 把debug tensor复制到cpu上
    u32* debug_tensor_cpu = (u32*)malloc(DEBUG_TENSOR_SIZE * sizeof(u32));
    cudaMemcpy(debug_tensor_cpu, debug_tensor, DEBUG_TENSOR_SIZE * sizeof(u32), cudaMemcpyDeviceToHost);

    // 打印debug tensor的内容
    for(u32 id_row=0;id_row<32;++id_row) {
        // 当前行的头指针
        u32* row_ptr = debug_tensor_cpu + id_row * 64;
        // 转换成main type的指针
        MainType* main_type_ptr = (MainType*)row_ptr;
        // 遍历打印每个数据
        for(u32 id_data=0;id_data<128;++id_data) {
            std::cout<<main_type_ptr[id_data]<<"\t";
        }
        std::cout<<std::endl;
    }
#endif
}