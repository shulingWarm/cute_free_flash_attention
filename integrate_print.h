#pragma once
#include<iostream>
#include"types.h"

class ThreadBlock {
public:
    u32 x;
    u32 y;
    u32 z;

    ThreadBlock(u32 x=0, u32 y=0, u32 z=0) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};

// 由cpu完成的attention计算
// 但仅仅是为了debug某些特定的计算块
template<class T>
void cpu_attention_qkt(T* query, T* key, T* value,
    // 打印时关注的线程块
    ThreadBlock thread_block,
    u32 id_warp,
    u32 seq_len,
    u32 head_num=32,
    u32 head_dim=128,
    // 在参数里面添加一个输出流用于打印log
    std::ostream& log_stream = std::cout
) {
    // 当前线程块负责计算的query的起始位置
    u32 query_offset = thread_block.x * 8*4*head_dim +
        thread_block.y*seq_len*head_dim +
        id_warp*head_dim*8;
    // query的头指针
    T* query_head = query + query_offset;

    // 初始化矩阵相乘结果的矩阵
    T output[8*16];
    
    // 执行矩阵相乘的计算
    for(u32 m=0;m<16;++m) {
        // 当前行的key头指针
        T* key_head = key + m*head_dim;
        for(u32 n=0;n<8;++n) {
            // 当前行的query
            T* curr_query_head = query_head + n*head_dim;
            T sum = T(0);
            for(u32 k=0;k<16;++k) {
                sum += curr_query_head[k] * key_head[k];
            }
            output[m*8+n] = sum;
        }
    }

    // 打印output结果
    log_stream << "query_offset: " << query_offset << std::endl;
    log_stream << "Attention QK^T output: " << std::endl;
    for(u32 m=0;m<16;++m) {
        for(u32 n=0;n<8;++n) {
            log_stream << output[m*8+n] << " ";
        }
        log_stream << std::endl;
    }
}