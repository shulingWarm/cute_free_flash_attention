#pragma once
#include<iostream>

// 由cpu完成的attention计算
// 但仅仅是为了debug某些特定的计算块
template<class T>
void cpu_attention_qkt(T* query, T* key, T* value,
    // 打印时关注的线程块
    
    // 在参数里面添加一个输出流用于打印log
    std::ostream& log_stream = std::cout
) {
    
}