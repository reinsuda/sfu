#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "utils.h"
#include <iostream>
#include <omp.h>

int main()
{
    // uint32_t tt = fp32_rcp(0x200001);
    // // rcp test
    // for (size_t src = 0; src < 0x100000000; src++)
    // {

    //     uint32_t src_rcp = (uint32_t)src;
    //     float float_input = *reinterpret_cast<float *>(&src_rcp);
    //     float g_f = 1 / float_input;
    //     uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);
    //     uint32_t rst = fp32_rcp(src_rcp);
    //     uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;
    //     if (diff >= 2)
    //     {
    //         if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
    //             std::cout << std::hex << "input 0x" << src_rcp << " gl: 0x" << g_u << " rst: 0x" << rst << " diff: " << diff << std::endl;
    //     }
    // }
    // sqrt_test();

    // // rcp test
    // for (size_t src = 0; src < 0x100000000; src++)
    // {

    //     uint32_t src_rcp = (uint32_t)src;
    //     float float_input = *reinterpret_cast<float *>(&src_rcp);
    //     float g_f = (float)pow(float_input, (double)0.5);
    //     uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);
    //     uint32_t rst = fp32_sqrt(src_rcp);
    //     uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;
    //     if (diff >= 2)
    //     {
    //         if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
    //             std::cout << std::hex << "input 0x" << src_rcp << " gl: 0x" << g_u << " rst: 0x" << rst << " diff: " << diff << std::endl;
    //     }
    // }

    // // rsqrt test
    // for (size_t src = 0; src < 0x100000000; src++)
    // {

    //     uint32_t src_rcp = (uint32_t)src;
    //     float float_input = *reinterpret_cast<float *>(&src_rcp);
    //     float g_f = 1.0f / sqrt(float_input);
    //     uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);
    //     uint32_t rst = fp32_rsq(src_rcp);
    //     uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;
    //     if (diff >= 4)
    //     {
    //         if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
    //             std::cout << std::hex << "input 0x" << src_rcp << " gl: 0x" << g_u << " rst: 0x" << rst << " diff: " << diff << std::endl;
    //     }
    // }
    // log2 test
    uint32_t max_ulp = 0;
    for (size_t src = 0; src < 0x100000000; src++)
    {

        uint32_t src_rcp = (uint32_t)src;
        float float_input = *reinterpret_cast<float *>(&src_rcp);
        float g_f = log2(float_input);
        uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);
        uint32_t rst = fp32_log2(src_rcp);
        uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;
        if (diff >= 7)
        {
            if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
                std::cout << std::hex << "input 0x" << src_rcp << " gl: 0x" << g_u << " rst: 0x" << rst << " diff: " << diff << std::endl;

            if (!fp32_is_nan(rst) && diff > max_ulp)
            {
                max_ulp = diff;
            }
        }
    }
    std::cout << "test done!!!" << std::endl;
    std::cout << std::hex << " max ulp: " << max_ulp << std::endl;
}