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
    // // rcp test
    // #pragma omp parallel for
    //     for (size_t src = 0; src < 0x100000000; src++)
    //     {

    //         uint32_t src_rcp = (uint32_t)src;
    //         float float_input = *reinterpret_cast<float *>(&src_rcp);
    //         float g_f = 1 / float_input;
    //         uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);
    //         uint32_t rst = fp32_rcp(src_rcp);
    //         uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;
    //         if (diff >= 2)
    //         {
    //             if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
    //                 std::cout << std::hex << "rcp input 0x" << src_rcp << " gl: 0x" << g_u << " rst: 0x" << rst << " diff: " << diff << std::endl;
    //         }
    //     }

    //     // // sqrt test
    // #pragma omp parallel for
    //     for (size_t src = 0; src < 0x100000000; src++)
    //     {

    //         uint32_t src_rcp = (uint32_t)src;
    //         float float_input = *reinterpret_cast<float *>(&src_rcp);
    //         float g_f = (float)pow(float_input, (double)0.5);
    //         uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);
    //         uint32_t rst = fp32_sqrt(src_rcp);
    //         uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;
    //         if (diff >= 2)
    //         {
    //             if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
    //                 std::cout << std::hex << "sqrt input 0x" << src_rcp << " gl: 0x" << g_u << " rst: 0x" << rst << " diff: " << diff << std::endl;
    //         }
    //     }

    //     // // rsqrt test
    //     uint32_t rsqrt_max_ulp = 0;
    // #pragma omp parallel for
    //     for (size_t src = 0; src < 0x100000000; src++)
    //     {

    //         uint32_t src_rcp = (uint32_t)src;
    //         float float_input = *reinterpret_cast<float *>(&src_rcp);
    //         float g_f = 1.0f / sqrt(float_input);
    //         uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);
    //         uint32_t rst = fp32_rsq(src_rcp);
    //         uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;
    //         if (diff >= 4)
    //         {
    //             if ((!fp32_is_nan(g_u) || !fp32_is_nan(rst)) && diff >= 5)
    //             {
    //                 std::cout << std::hex << "rsqrt input 0x" << src_rcp << " gl: 0x" << g_u << " rst: 0x" << rst << " diff: " << diff << std::endl;
    //             }
    //             if (!fp32_is_nan(rst) && diff > rsqrt_max_ulp)
    //             {
    //                 rsqrt_max_ulp = diff;
    //             }
    //         }
    //     }
    //     std::cout << std::hex << "rsqrt max ulp: " << rsqrt_max_ulp << std::endl;
    //     // log2 test
    //     uint32_t max_ulp = 0;
    //     for (size_t src = 0; src < 0x100000000; src++)
    //     {

    //         uint32_t src_rcp = (uint32_t)src;
    //         float float_input = *reinterpret_cast<float *>(&src_rcp);
    //         float g_f = log2(float_input);
    //         uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);
    //         uint32_t rst = fp32_log2(src_rcp);
    //         uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;
    //         if (diff >= 8)
    //         {
    //             // if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
    //             //     std::cout << std::hex << "log2 input 0x" << src_rcp << " gl: 0x" << g_u << " rst: 0x" << rst << " diff: " << diff << std::endl;

    //             if (!fp32_is_nan(rst) && diff > max_ulp)
    //             {
    //                 max_ulp = diff;
    //             }
    //         }
    //     }

    //     std::cout << std::hex << "log2 max ulp: " << max_ulp << std::endl;

    //     // test sin
    //     uint32_t sin_max_ulp = 0;
    //     uint32_t sin_max_input = 0;
    // #pragma omp parallel for
    //     for (size_t src = 0; src < 0x100000000; src++)
    //     {
    //         uint32_t sfu_input = (uint32_t)src;
    //         float src_f = *reinterpret_cast<float *>(&sfu_input);

    //         // Golden 模型计算 sin(x * 2pi)
    //         double int_part;
    //         double frac_part = modf((double)src_f, &int_part);
    //         float g_f = (float)sin(frac_part * 2.0 * M_PI);
    //         uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);

    //         // SFU 硬件模型计算
    //         uint32_t rst = fp32_sin(sfu_input, false);

    //         uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;

    //         if (diff >= 0x10 && (frac_part != 0.5) && frac_part != -0.5) //  diff != 0x5af2cece 是因为刚好等于π的时候golden会有问题 diff != 0xa50d3132(-π)
    //         {
    //             if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
    //             {
    //                 // std::cout << std::hex << "sin input: 0x" << sfu_input
    //                 //           << " gl: 0x" << g_u
    //                 //           << " rst: 0x" << rst
    //                 //           << " diff: " << diff << std::endl;
    // // assert(false);
    // #pragma omp critical
    //                 {
    //                     if (!fp32_is_nan(rst) && diff > sin_max_ulp)
    //                     {
    //                         sin_max_ulp = diff;
    //                         sin_max_input = sfu_input;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     //(0 ~1)24个ulp
    //     std::cout << std::hex << "sin max ulp: " << sin_max_ulp << " input: " << sin_max_input << std::endl;

    //     // test cos
    //     uint32_t cos_max_ulp = 0;
    //     uint32_t cos_max_input = 0;
    // #pragma omp parallel for
    //     for (size_t src = 0; src < 0x100000000; src++)
    //     {
    //         uint32_t sfu_input = (uint32_t)src;
    //         float src_f = *reinterpret_cast<float *>(&sfu_input);

    //         // Golden 模型计算 sin(x * 2pi)
    //         double int_part;
    //         double frac_part = modf((double)src_f, &int_part);
    //         float g_f = (float)cos(frac_part * 2.0 * M_PI);
    //         uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);

    //         // SFU 硬件模型计算
    //         uint32_t rst = fp32_cos(sfu_input, false);

    //         uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;

    //         if (diff >= 0x10 && (abs(frac_part) != 0.25) && abs(frac_part) != 0.75) //
    //         {
    //             if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
    //             {
    //                 // std::cout << std::hex << "cos input: 0x" << sfu_input
    //                 //           << " gl: 0x" << g_u
    //                 //           << " rst: 0x" << rst
    //                 //           << " diff: " << diff << std::endl;
    // // assert(false);
    // #pragma omp critical
    //                 {
    //                     if (!fp32_is_nan(rst) && diff > cos_max_ulp)
    //                     {
    //                         cos_max_ulp = diff;
    //                         cos_max_input = sfu_input;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //(0 ~1)24个ulp
    // std::cout << std::hex << "cos max ulp: " << cos_max_ulp << " input: " << cos_max_input << std::endl;

    std::cout << "--- Start fp32_exp2 exhaustive test ---" << std::endl;

    uint32_t exp2_max_ulp = 0;
    uint32_t exp2_error_count = 0;
#pragma omp parallel for
    for (size_t src = 0; src < 0x100000000; src++)
    {
        uint32_t src_hex = (uint32_t)src;
        float float_input = *reinterpret_cast<float *>(&src_hex);

        // 1. 获取 CPU 双精度计算的 Golden 结果
        float g_f = std::exp2(float_input);
        uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);

        // 2. 获取你的硬件仿真结果
        uint32_t rst = fp32_exp2(src_hex);

        // ==========================================
        // 💡 硬件行为对齐滤镜 (Filters)
        // ==========================================

        // 滤镜 A: NaN 屏蔽。
        // CPU 产生的 NaN 和硬件产生的 NaN 只要阶码全为 1 且尾数非 0 就是合法的，
        // 它们的尾数 Payload 可能不同，相减会产生巨大 Diff，直接跳过不比对。
        bool is_in_nan = ((src_hex & 0x7F800000) == 0x7F800000) && ((src_hex & 0x007FFFFF) != 0);
        if (is_in_nan)
            continue;

        // 滤镜 B: CPU 非规格化数冲刷到零 (Flush-to-Zero, FTZ)。
        // 硬件通常不保留非规格化数，如果 CPU 算出的阶码是 0，强行刷成 0

        // ==========================================

        // 3. 计算 ULP Diff
        uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;

        // 4. 统计与报错逻辑 (假设我们容忍最大 4 ULP 的误差)
        if (diff > exp2_max_ulp)
        {
            exp2_max_ulp = diff;
        }

        // 打印出超过宽容度的异常值，方便 Debug
        if (diff > 4)
        {
#pragma omp critical
            {
                exp2_error_count++;
                // 为了防止刷屏，只打印前 20 个错误
                if (exp2_error_count <= 20)
                {
                    std::cout << std::hex
                              << "exp2 input: 0x" << src_hex
                              << " (" << float_input << ")"
                              << " | gl: 0x" << g_u
                              << " | rst: 0x" << rst
                              << std::dec << " | diff: " << diff
                              << std::endl;
                }
            }
        }
    }

    std::cout << std::dec << "exp2 errors (>4 ULP): " << exp2_error_count << std::endl;
    std::cout << std::hex << "exp2 max ulp: " << exp2_max_ulp << std::endl;
    std::cout << "---------------------------------------" << std::endl;
}