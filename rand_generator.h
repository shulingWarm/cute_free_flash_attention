#pragma once
#include <random>
#include <iostream>

class UniformRandomGenerator {
private:
    std::random_device rd;  // 用于生成随机种子
    std::mt19937 generator; // Mersenne Twister 生成器

public:
    // 构造函数：初始化生成器
    UniformRandomGenerator() : generator(rd()) {}

    // 生成指定范围内的浮点数
    double generate(double min, double max) {
        // 检查参数有效性
        if (min >= max) {
            std::cout<<"min more than max"<<std::endl;
            throw -1;
        }
        
        // 创建均匀分布对象
        std::uniform_real_distribution<double> distribution(min, max);
        
        // 生成随机数
        return distribution(generator);
    }

    // 可选：重新设置随机数生成器的种子
    void seed(unsigned int seed_value) {
        generator.seed(seed_value);
    }
};



// 用于随机初始化CPU矩阵的函数
template<class T>
void init_matrix(T *matrix, int rows, int cols, UniformRandomGenerator& generator) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (T)generator.generate(-0.5, 0.5);
        }
        std::cout<<std::endl;
    }
}

// 通过输入数据长度来初始化tensor
template<class T>
void init_matrix_with_length(T *tensor, int length, UniformRandomGenerator& generator) {
    for (int i = 0; i < length; i++) {
        tensor[i] = (T)generator.generate(-0.5, 0.5);
    }
}