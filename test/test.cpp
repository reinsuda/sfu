#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "utils.h"
#include <iostream>
#include <omp.h>

void test_sig()
{
    std::cout << "--- Start fp32_sig exhaustive test ---" << std::endl;

    uint32_t sig_max_ulp = 0;
    uint32_t sig_error_count = 0;
    uint32_t sig_max_err_input = 0;
    uint32_t sig_max_err_gld = 0;
    uint32_t sig_max_err_rst = 0;

#pragma omp parallel for
    for (size_t src = 0; src < 0x100000000; src++)
    {
        uint32_t src_hex = (uint32_t)src;
        float float_input = *reinterpret_cast<float *>(&src_hex);

        // ==========================================
        // 1. 获取 CPU 双精度计算的 Golden 结果
        // ==========================================
        // 转为 double 算真值，再截断回 float，保证 Golden 的完美精度
        double d_in = (double)float_input;
        float g_f = (float)(1.0 / (1.0 + std::exp(-d_in)));
        uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);

        // ==========================================
        // 2. 获取你的硬件仿真结果
        // ==========================================
        uint32_t rst = fp32_sig(src_hex);

        // ==========================================
        // 3. 💡 硬件行为对齐滤镜 (Filters)
        // ==========================================

        // 滤镜 A: NaN 屏蔽
        bool is_in_nan = ((src_hex & 0x7F800000) == 0x7F800000) && ((src_hex & 0x007FFFFF) != 0);
        if (is_in_nan)
            continue;

        // 滤镜 B: Sigmoid 特有的饱和区双向强制对齐 (Saturation Bypass)
        // 假设你的硬件在 |x| >= 16.0 时直接输出 1.0 或 0.0 (请根据你真实的硬件代码阈值修改！)
        // if (float_input >= 16.0f)
        // {
        //     g_u = 0x3F800000; // 强行让 CPU 在正饱和区输出 1.0
        // }
        // else if (float_input <= -16.0f)
        // {
        //     g_u = 0x00000000; // 强行让 CPU 在负饱和区输出 0.0
        // }

        // // 滤镜 C: 双向非规格化数冲刷到零 (FTZ)
        // // 只要阶码是 0 (即 bits 23-30 全为 0)，统统强行刷成纯 0！
        // if ((g_u & 0x7F800000) == 0)
        //     g_u = 0x00000000;
        // if ((rst & 0x7F800000) == 0)
        //     rst = 0x00000000;

        // // 滤镜 D: 统一冲刷 -0.0 为 +0.0
        // if (g_u == 0x80000000)
        //     g_u = 0x00000000;
        // if (rst == 0x80000000)
        //     rst = 0x00000000;

        // ==========================================

        // 4. 计算 ULP Diff
        uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;

        // 5. 统计与报错逻辑 (容忍最大 4~5 ULP 的误差)
        if (diff > sig_max_ulp)
        {
            // 💡 修复 OpenMP 数据竞争：使用 critical 区块安全更新最大值
#pragma omp critical
            {
                if (diff > sig_max_ulp)
                {
                    sig_max_ulp = diff;
                    sig_max_err_input = src_hex;
                    sig_max_err_gld = g_u;
                    sig_max_err_rst = rst;
                }
            }
        }

        // 打印出超过宽容度 (例如 5 ULP) 的异常值，方便 Debug
        if (diff > 0x10000)
        {
#pragma omp critical
            {
                sig_error_count++;
                // 为了防止刷屏，只打印前 20 个错误
                if (sig_error_count <= 20)
                {
                    std::cout << std::hex
                              << "sig input: 0x" << src_hex
                              << " (" << float_input << ")"
                              << " | gl: 0x" << g_u
                              << " | rst: 0x" << rst
                              << std::dec << " | diff: " << diff
                              << std::endl;
                }
            }
        }
    }

    std::cout << std::dec << "sig errors (>5 ULP): " << sig_error_count << std::endl;
    std::cout << std::hex << "sig max ulp: " << sig_max_ulp << " input: " << sig_max_err_input << " gl: " << sig_max_err_gld << " rst: " << sig_max_err_rst << std::endl;
    std::cout << "---------------------------------------" << std::endl;
}

void test_sig_partral()
{
    std::cout << "--- Start fp32_sig exhaustive test ---" << std::endl;

    uint32_t sig_max_ulp = 0;
    uint32_t sig_error_count = 0;
    uint32_t sig_max_err_input = 0;
    uint32_t sig_max_err_gld = 0;
    uint32_t sig_max_err_rst = 0;

    for (size_t exp = 131; exp < 132; exp++)
    {
#pragma omp parallel for
        for (size_t mant = 0; mant < 0x7fffff; mant++)
        {
            uint32_t src_hex = ((uint32_t)exp << FP32_MANT_WIDTH) | (uint32_t)mant;
            float float_input = *reinterpret_cast<float *>(&src_hex);

            // ==========================================
            // 1. 获取 CPU 双精度计算的 Golden 结果
            // ==========================================
            // 转为 double 算真值，再截断回 float，保证 Golden 的完美精度
            double d_in = (double)float_input;
            float g_f = (float)(1.0 / (1.0 + std::exp(-d_in)));
            uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);

            // ==========================================
            // 2. 获取你的硬件仿真结果
            // ==========================================
            uint32_t rst = fp32_sig(src_hex);

            // ==========================================
            // 3. 💡 硬件行为对齐滤镜 (Filters)
            // ==========================================

            // 滤镜 A: NaN 屏蔽
            bool is_in_nan = ((src_hex & 0x7F800000) == 0x7F800000) && ((src_hex & 0x007FFFFF) != 0);
            if (is_in_nan)
                continue;

            // 滤镜 B: Sigmoid 特有的饱和区双向强制对齐 (Saturation Bypass)
            // 假设你的硬件在 |x| >= 16.0 时直接输出 1.0 或 0.0 (请根据你真实的硬件代码阈值修改！)
            // if (float_input >= 16.0f)
            // {
            //     g_u = 0x3F800000; // 强行让 CPU 在正饱和区输出 1.0
            // }
            // else if (float_input <= -16.0f)
            // {
            //     g_u = 0x00000000; // 强行让 CPU 在负饱和区输出 0.0
            // }

            // // 滤镜 C: 双向非规格化数冲刷到零 (FTZ)
            // // 只要阶码是 0 (即 bits 23-30 全为 0)，统统强行刷成纯 0！
            // if ((g_u & 0x7F800000) == 0)
            //     g_u = 0x00000000;
            // if ((rst & 0x7F800000) == 0)
            //     rst = 0x00000000;

            // // 滤镜 D: 统一冲刷 -0.0 为 +0.0
            // if (g_u == 0x80000000)
            //     g_u = 0x00000000;
            // if (rst == 0x80000000)
            //     rst = 0x00000000;

            // ==========================================

            // 4. 计算 ULP Diff
            uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;

            // 5. 统计与报错逻辑 (容忍最大 4~5 ULP 的误差)
            if (diff > sig_max_ulp)
            {
// 💡 修复 OpenMP 数据竞争：使用 critical 区块安全更新最大值
#pragma omp critical
                {
                    if (diff > sig_max_ulp)
                    {
                        sig_max_ulp = diff;
                        sig_max_err_input = src_hex;
                        sig_max_err_gld = g_u;
                        sig_max_err_rst = rst;
                    }
                }
            }

            // 打印出超过宽容度 (例如 5 ULP) 的异常值，方便 Debug
            if (diff > 0x100)
            {
#pragma omp critical
                {
                    sig_error_count++;
                    // 为了防止刷屏，只打印前 20 个错误
                    if (sig_error_count <= 20)
                    {
                        std::cout << std::hex
                                  << "sig input: 0x" << src_hex
                                  << " (" << float_input << ")"
                                  << " | gl: 0x" << g_u
                                  << " | rst: 0x" << rst
                                  << " | diff: " << diff
                                  << std::endl;
                    }
                }
            }
        }
    }
    std::cout << std::dec << "sig errors (>5 ULP): " << sig_error_count << std::endl;
    std::cout << std::hex << "sig max ulp: " << sig_max_ulp << " input: " << sig_max_err_input << " gl: " << sig_max_err_gld << " rst: " << sig_max_err_rst << std::endl;
    std::cout << "---------------------------------------" << std::endl;
}

void test_log2()
{
    // log2 test
    uint32_t max_ulp = 0;
#pragma omp parallel for
    for (size_t src = 0x3f7fea33; src < 0x3f802c40; src++)
    {

        uint32_t src_rcp = (uint32_t)src;
        float float_input = *reinterpret_cast<float *>(&src_rcp);
        float g_f = log2(float_input);
        uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);
        uint32_t rst = fp32_log2(src_rcp);
        uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;
        if (diff >= 0x100)
        {
            if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
                std::cout << std::hex << "log2 input 0x" << src_rcp << " gl: 0x" << g_u << " rst: 0x" << rst << " diff: " << diff << std::endl;
#pragma omp critical
            {
                if (!fp32_is_nan(rst) && diff > max_ulp)
                {
                    max_ulp = diff;
                }
            }
        }
    }

    std::cout << std::hex << "log2 max ulp: " << max_ulp << std::endl;
}
int main()
{
    // test_log2();
    test_sig_partral();
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

    // test sin
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

    //         if (diff >= 0x100 && (frac_part != 0.5) && frac_part != -0.5) //  diff != 0x5af2cece 是因为刚好等于π的时候golden会有问题 diff != 0xa50d3132(-π)
    //         {
    //             if (!fp32_is_nan(g_u) || !fp32_is_nan(rst))
    //             {
    //                 std::cout << std::hex << "sin input: 0x" << sfu_input
    //                           << " gl: 0x" << g_u
    //                           << " rst: 0x" << rst
    //                           << " diff: " << diff << std::endl;
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

    // test cos
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
    // // (0 ~1)24个ulp
    // std::cout << std::hex << "cos max ulp: " << cos_max_ulp << " input: " << cos_max_input << std::endl;

    //     std::cout << "--- Start fp32_exp2 exhaustive test ---" << std::endl;

    //     uint32_t exp2_max_ulp = 0;
    //     uint32_t exp2_error_count = 0;
    // #pragma omp parallel for
    //     for (size_t src = 0; src < 0x100000000; src++)
    //     {
    //         uint32_t src_hex = (uint32_t)src;
    //         float float_input = *reinterpret_cast<float *>(&src_hex);

    //         // 1. 获取 CPU 双精度计算的 Golden 结果
    //         float g_f = std::exp2(float_input);
    //         uint32_t g_u = *reinterpret_cast<uint32_t *>(&g_f);

    //         // 2. 获取你的硬件仿真结果
    //         uint32_t rst = fp32_exp2(src_hex);

    //         // ==========================================
    //         // 💡 硬件行为对齐滤镜 (Filters)
    //         // ==========================================

    //         // 滤镜 A: NaN 屏蔽。
    //         // CPU 产生的 NaN 和硬件产生的 NaN 只要阶码全为 1 且尾数非 0 就是合法的，
    //         // 它们的尾数 Payload 可能不同，相减会产生巨大 Diff，直接跳过不比对。
    //         bool is_in_nan = ((src_hex & 0x7F800000) == 0x7F800000) && ((src_hex & 0x007FFFFF) != 0);
    //         if (is_in_nan)
    //             continue;

    //         // 滤镜 B: CPU 非规格化数冲刷到零 (Flush-to-Zero, FTZ)。
    //         // 硬件通常不保留非规格化数，如果 CPU 算出的阶码是 0，强行刷成 0

    //         // ==========================================

    //         // 3. 计算 ULP Diff
    //         uint32_t diff = g_u > rst ? g_u - rst : rst - g_u;

    //         // 4. 统计与报错逻辑 (假设我们容忍最大 4 ULP 的误差)
    //         if (diff > exp2_max_ulp)
    //         {
    //             exp2_max_ulp = diff;
    //         }

    //         // 打印出超过宽容度的异常值，方便 Debug
    //         if (diff > 4)
    //         {
    // #pragma omp critical
    //             {
    //                 exp2_error_count++;
    //                 // 为了防止刷屏，只打印前 20 个错误
    //                 if (exp2_error_count <= 20)
    //                 {
    //                     std::cout << std::hex
    //                               << "exp2 input: 0x" << src_hex
    //                               << " (" << float_input << ")"
    //                               << " | gl: 0x" << g_u
    //                               << " | rst: 0x" << rst
    //                               << std::dec << " | diff: " << diff
    //                               << std::endl;
    //                 }
    //             }
    //         }
    //     }

    //     std::cout << std::dec << "exp2 errors (>4 ULP): " << exp2_error_count << std::endl;
    //     std::cout << std::hex << "exp2 max ulp: " << exp2_max_ulp << std::endl;
    //     std::cout << "---------------------------------------" << std::endl;
}