#include <chrono>
#include <numeric>
#include <functional>
#include <algorithm>
#include <vector>
#include <iostream>
//#include "crc32c.h"


int64_t hash_function_0(int64_t a, int64_t b) {
    return a ^ ((b << 8) | (b >> 8));
}

int64_t hash_function_1(int64_t a, int64_t b) {
    return (a * 106039 + b) & 0xffff;
}

// best among 0~4
int64_t hash_function_2(int64_t a, int64_t b) {
    return (a * 9824516537u + b);
}

int64_t cantor_pairing(int64_t a, int64_t b) {
    return (a + b) * (a + b + 1) / 2 + b;
}

int64_t hash_function_3(int64_t a, int64_t b) {
    int64_t paired_num = cantor_pairing(a, b);
    paired_num ^= paired_num >> 15;
    paired_num ^= paired_num << 17;
    return paired_num ^ (paired_num >> 5);
}

int64_t hash_function_4(int64_t a, int64_t b) {
    int64_t paired_num = cantor_pairing(a, b);
    paired_num = paired_num * 9824516537u;
    return paired_num;

}
/*
int64_t hash_function_5(int64_t a, int64_t b) {
    int32_t temp[4];
    temp[0] = a & 0xffffffffu;
    temp[1] = (a >> 32) & 0xffffffffu;
    temp[2] = b & 0xffffffffu;
    temp[3] = (b >> 32) & 0xffffffffu;
    return crc32c((char*)temp, 4);
}*/

int64_t hash_function_6(int64_t a, int64_t b) {
    return (a * 9824516537u + b * 57857966300227u) % 117130198221199u;
}
std::vector<std::pair<int64_t, int64_t>> make_input(int64_t height, int64_t width) {
    std::vector<std::pair<int64_t, int64_t>> result;
    for (int64_t i = 0; i < height; ++i) {
        for (int64_t j = 0; j < width; ++j) {
            result.push_back({i, j});
        }
    }
    return result;
}

double get_variance(const std::vector<int64_t>& occurance) {
    int64_t total_sum = std::accumulate(occurance.begin(), occurance.end(), 0);
    double mean = static_cast<double>(total_sum) / occurance.size();
    double error_square_sum = 0.0;
    for (int64_t element : occurance) {
        double error = static_cast<double>(element) - mean;
        error_square_sum += error * error;
    }    
    return error_square_sum / occurance.size();
}

int main() {
    std::vector<std::pair<int64_t, int64_t>> input_sizes;

    input_sizes.push_back({100, 128});
    input_sizes.push_back({200, 128});
    input_sizes.push_back({500, 128});
    input_sizes.push_back({1000, 128});
    input_sizes.push_back({2000, 128});
    input_sizes.push_back({5000, 128});
    input_sizes.push_back({10000, 128});

    std::vector<std::function<int64_t(int64_t, int64_t)>> hash_functions;
    hash_functions.push_back(hash_function_0);
    hash_functions.push_back(hash_function_1);
    hash_functions.push_back(hash_function_2);
    hash_functions.push_back(hash_function_3);
    hash_functions.push_back(hash_function_4);
    //hash_functions.push_back(hash_function_5);
    hash_functions.push_back(hash_function_6);

    std::vector<std::vector<std::pair<int64_t, int64_t>>> input_groups;
    for (auto& input_size : input_sizes) {
        input_groups.push_back(make_input(input_size.first, input_size.second));
    }

    double compression_rates[] = {1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0 / 1.0};

    for (double compression_rate : compression_rates) {
        std::cout << "testing for compression rate " << compression_rate << std::endl;
        for (auto& input_group: input_groups) {
            std::cout << "input_size is " << input_group.size() << std::endl;
            int64_t compressed_size = 
                static_cast<int64_t>(static_cast<double>(input_group.size()) * compression_rate);
            for (size_t hash_idx = 0; hash_idx < hash_functions.size(); ++hash_idx) {
                std::vector<int64_t> hashed_vector(compressed_size, 0);
                const auto start_time = std::chrono::high_resolution_clock::now();
                for (auto& input_pair : input_group) {
                    uint64_t hashed_value = hash_functions[hash_idx](input_pair.first, input_pair.second);
                    hashed_value %= compressed_size;
                    hashed_vector[hashed_value]++;
                }
                const auto end_time = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double, std::milli> running_time = end_time - start_time;
                double variance = get_variance(hashed_vector);
                std::cout << "hash func" << hash_idx << ": time=" << running_time.count() << " var=" << variance << ", ";
                
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    return 0;
}