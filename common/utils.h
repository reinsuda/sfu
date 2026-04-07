#pragma once

#include <vector>
#include <queue>
#include <string>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cstdint>

typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;

#define PI 3.1415927

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define BIT_N(value, pos) ((value >> (pos)) & 0x1)
#define N_BIT_1(N) (((uint64_t)1 << (N)) - 1)

#define FP32_NAN 0x7FC00000
#define FP32_INF 0x7F800000
// these are fixed !!!
#define FP32_MANT_WIDTH 23
#define FP32_EXP_WIDTH 8
#define FP16_MANT_WIDTH 5
#define FP16_EXP_WIDTH 10

// the config for RCP
#define RCP_TABLE_BIT_WIDTH 7                               // all tables' item number reduce to 1/8 of original
const uint32_t RCP_TABLE_ITEM = (1 << RCP_TABLE_BIT_WIDTH); // all tables' item number reduce to 1/8 of original

#define LOG2_TABLE_BIT_WIDTH 7                                // all tables' item number reduce to 1/8 of original
const uint32_t LOG2_TABLE_ITEM = (1 << LOG2_TABLE_BIT_WIDTH); // all tables' item number reduce to 1/8 of original

// the config for EXP
#define EXP_TABLE_BIT_WIDTH 4                     // all tables' item number reduce to 1/8 of original
#define EXP_TABLE_ITEM (1 << EXP_TABLE_BIT_WIDTH) // all tables' item number reduce to 1/8 of original

// the config for EXP
#define LOG_TABLE_BIT_WIDTH 4                     // all tables' item number reduce to 1/8 of original
#define LOG_TABLE_ITEM (1 << LOG_TABLE_BIT_WIDTH) // all tables' item number reduce to 1/8 of original

// the config for EXP
#define RSQ_TABLE_BIT_WIDTH 4                     // all tables' item number reduce to 1/8 of original
#define RSQ_TABLE_ITEM (1 << RSQ_TABLE_BIT_WIDTH) // all tables' item number reduce to 1/8 of original

// the config for RCP
#define SQRT_TABLE_BIT_WIDTH 6                      // all tables' item number reduce to 1/8 of original
#define SQRT_TABLE_ITEM (1 << SQRT_TABLE_BIT_WIDTH) // all tables' item number reduce to 1/8 of original

// a gloable vector record the sign of A B C of each table
static std::vector<std::vector<uint32_t>> sign_record = {{0, 1, 0}, {0, 1, 1}};

enum functions
{
    reciprocal,
    squ_root,
    reci_squ_root,
    exponent2,
    logarithm2,
    sine,
    cosine,
    sigmoid,
};

typedef struct
{
    int64_t C0;
    int64_t C1;
    int64_t C2;
    uint32_t X1; // [MAX_M]
    uint32_t X2; // [MAN_NUM - MIN_M]
} Coeff;

typedef struct
{
    uint32_t c0;     // Offset
    uint32_t c1_abs; // Absolute value of 1st order coeff
    uint32_t c2;     // 2nd order coeff (Two's Complement)
} rcp_entry_t;

struct TABLE
{
    double pos = 0;
    uint64_t A = 0;
    uint64_t B = 0;
    uint64_t C = 0;
};

struct Precision
{
    uint32_t A_pre = 24;
    uint32_t B_pre = 18;
    uint32_t C_pre = 8;
};

static bool fp32_is_nan(uint32_t input)
{
    uint32_t exp;
    uint32_t man;
    exp = (input >> 23) & 0xFF;
    man = input & 0x7FFFFF;
    return ((exp == 0xFF) && (man != 0));
}
static bool fp32_is_inf(uint32_t input)
{
    uint32_t exp;
    uint32_t man;
    exp = (input >> 23) & 0xFF;
    man = input & 0x7FFFFF;
    return ((exp == 0xFF) && (man == 0));
}
static bool fp32_is_zero(uint32_t input)
{
    uint32_t sign = input & 0x80000000;
    uint32_t nonsign = input ^ sign;
    return (nonsign == 0);
}

// 辅助函数：将 64位无符号整数 按照 RTNE (向偶数舍入) 规则进行移位
static uint32_t ShiftRightAndRound(uint64_t val, int shift_amt)
{
    if (shift_amt == 0)
        return (uint32_t)val;

    // 如果需要左移（说明有效位比较低，直接左移即可，无需舍入）
    if (shift_amt < 0)
    {
        return (uint32_t)(val << -shift_amt);
    }

    // 处理极端的右移下溢情况
    if (shift_amt >= 64)
    {
        if (shift_amt == 64)
        {
            uint32_t g = (val >> 63) & 1;
            uint32_t s = (val & 0x7FFFFFFFFFFFFFFFULL) != 0;
            // 结果为 0 (偶数)，仅当 Guard=1 且 Sticky=1 时才会向上进位变成 1
            if (g && s)
                return 1;
        }
        return 0;
    }

    // --- 标准的 RTNE 舍入逻辑 ---
    uint32_t g = (val >> (shift_amt - 1)) & 1; // Guard bit (被移出的最高位)
    uint64_t mask = (1ULL << (shift_amt - 1)) - 1;
    uint32_t s = (val & mask) != 0;                 // Sticky bit (余下的所有低位是否有1)
    uint32_t result = (uint32_t)(val >> shift_amt); // 移位后的纯尾数

    // RTNE: 向最接近的数舍入，如果正好在中间(0.5)，则向偶数舍入
    // 进位条件：Guard 为 1 且 (Sticky 为 1 或者 当前结果为奇数)
    if (g && (s || (result & 1)))
    {
        result++;
    }

    return result;
}

// 主算法：定点数转 FP32
static uint32_t NormalizeToFP32(uint64_t table_res, int exp, uint32_t dec_bits)
{
    // 1. 提取符号位，如果原值是负数(补码)，则转为绝对值
    uint32_t sign = (table_res >> 63) & 1;
    if (sign)
    {
        table_res = ~table_res + 1;
    }

    // 2. 特例：如果定点数完全为 0
    if (table_res == 0)
    {
        return sign << 31; // 保持正负 0 的符号
    }

    // 3. 寻找最高有效位位置 (pos)
    // 实际工程中可替换为硬件指令如 __builtin_clzll(table_res) 以极致提升性能
    uint32_t pos = 63 - __builtin_clzll(table_res);
    // while ((table_res & (1ULL << pos)) == 0)
    // {
    //     pos--;
    // }

    // 4. 计算理想状态下的 FP32 阶码
    // 数学依据: FP32_Exp = 传入的偏置Exp + 当前最高有效位 - 定点数的小数位数
    int raw_exp = exp + (int)pos - (int)dec_bits;

    uint32_t mantissa = 0;

    // 5. 核心分支：规格化数 (Normal) 与 非规格化数 (Subnormal) 分别处理
    if (raw_exp > 0)
    {
        // ========== 规格化数 (Normal Number) ==========

        // 目标是将 MSB (pos) 恰好对齐到尾数的第 23 位 (隐藏位)
        int shift_amt = (int)pos - 23;
        mantissa = ShiftRightAndRound(table_res, shift_amt);

        // 检查舍入是否导致了尾数溢出进位 (例如 0xFFFFFF 进位变成了 0x1000000)
        // 这个判断优雅地取代了你原来代码里的 == 0xffffff 的繁琐判断
        if (mantissa == (1 << 24))
        {
            mantissa >>= 1; // 尾数右移归位
            raw_exp++;      // 指数补偿进位
        }

        // 检查是否上溢到了无穷大 (Infinity)
        if (raw_exp >= 255)
        {
            return (sign << 31) | 0x7F800000;
        }

        // 组装最终结果: 剔除第 23 位的隐藏位 1 (使用 & 0x7FFFFF)，拼装符号和阶码
        return (sign << 31) | ((uint32_t)raw_exp << 23) | (mantissa & 0x7FFFFF);
    }
    else
    {
        // ========== 非规格化数 (Subnormal Number) / 下溢出 ==========

        // 非规格化数在 IEEE754 中的实际偏置指数被视为 1 (即使二进制阶码全填 0)
        // 它的 MSB 必须对齐到其真实的量级，所以所需的右移量是一个固定公式：
        int shift_amt = (int)dec_bits - exp - 22;

        mantissa = ShiftRightAndRound(table_res, shift_amt);

        // 检查舍入后的非规格化数是否满溢，恰好变成了最小的规格化数
        if (mantissa >= (1 << 23))
        {
            // 指数变为 1，尾数变为 0
            return (sign << 31) | (1 << 23);
        }

        // 组装非规格化数结果 (指数位全为 0，尾数不剔除任何位)
        return (sign << 31) | mantissa;
    }
}

static uint64_t fix_multi(uint64_t A, uint32_t B, uint32_t C, uint64_t delta,
                          uint32_t A_shit_w, uint32_t B_shift_w, uint32_t c_shift_w, uint32_t a_sign, uint32_t b_sign, uint32_t c_sign, uint32_t moreShift = 0)
{
    uint64_t B_mul_del = delta * B;             // 0.23 * 0.15 -> 0.30
    uint64_t C_mul_del_del = delta * delta * C; // 0.23 * 0.23 * 0.8 -> 0.54

    uint64_t tempA = A << A_shit_w;
    // 💡 修复1：使用 <= 防止下溢出
    B_mul_del = (moreShift <= B_shift_w) ? (B_mul_del << (B_shift_w - moreShift))
                                         : (B_mul_del >> (moreShift - B_shift_w));

    uint32_t c_more_shift = moreShift << 1;
    // 💡 修复2：既然减数是 c_more_shift，判断条件必须是 c_more_shift <= c_shift_w！
    C_mul_del_del = (c_more_shift <= c_shift_w) ? (C_mul_del_del << (c_shift_w - c_more_shift))
                                                : (C_mul_del_del >> (c_more_shift - c_shift_w));
    tempA = a_sign ? (~tempA) + 1 : tempA;
    B_mul_del = b_sign ? ~B_mul_del + 1 : B_mul_del;
    C_mul_del_del = c_sign ? ~C_mul_del_del + 1 : C_mul_del_del;
    return tempA + B_mul_del + C_mul_del_del;
}

// // A + B*delta + C*delta*delta
// static uint32_t SQRT_fix_multi(uint32_t A, uint32_t B, uint32_t C, uint32_t delta,
//                                uint32_t wid_A, uint32_t wid_B, uint32_t wid_C, uint32_t table_idx,
//                                uint32_t wid_delta = FP32_MANT_WIDTH) // a + bx + cx^2
// {
//     // 25(A, 0.14) 19(B, 0.8) 13(C, 0.3) 11(delta, mantissa[15:0])
//     // A's sign: 0, B's sign: 1, C's sign: 0
//     // delta is with a sign on table0 ? [15:15] : [16:16]
//     uint32_t sign_pos = table_idx == 0 ? FP32_MANT_WIDTH - SQRT_TABLE_BIT_WIDTH - 1
//                                        : FP32_MANT_WIDTH - SQRT_TABLE_BIT_WIDTH;
//     uint64_t sign_delta = (delta >> sign_pos) & 0x1 ? 0 : 1;
//     uint64_t ABS_delta = delta & ((1 << sign_pos) - 1);

//     uint64_t wid_frac_BXdel = wid_B + wid_delta;
//     uint64_t wid_frac_CXdel = wid_C + wid_delta + wid_delta;
//     uint64_t sign_A = 0;
//     uint64_t sign_B = 0;
//     uint64_t sign_C = 1;
//     uint64_t max_width = max(max(wid_frac_BXdel, wid_frac_CXdel), wid_A); // max width
//     return fix_multi((uint64_t)A, B, C, ABS_delta, max_width - wid_A, max_width - wid_frac_BXdel, max_width - wid_frac_CXdel, sign_A, sign_delta ^ sign_B, sign_C);
// }

// 修正后的 SQRT_fix_multi，彻底告别硬编码
static uint64_t SQRT_fix_multi(uint32_t A, uint32_t B, uint32_t C, uint32_t delta,
                               uint32_t wid_A, uint32_t wid_B, uint32_t wid_C, uint32_t table_idx,
                               uint32_t wid_delta = FP32_MANT_WIDTH)
{
    // 1. 动态计算当前区间的中心点偏移
    // 例如：128段表 (7 bit)，剩余 16 bit，中心点就在 1 << 15
    uint32_t frac_bits = FP32_MANT_WIDTH - SQRT_TABLE_BIT_WIDTH;
    uint32_t center_offset = 1 << (frac_bits - 1);

    // 2. 以中心点为 0，计算有符号的 delta
    int32_t signed_delta = (int32_t)delta - center_offset;

    // 3. 提取符号和绝对值
    uint64_t sign_delta = (signed_delta < 0) ? 1 : 0;
    uint64_t ABS_delta = sign_delta ? (~signed_delta & ((1 << frac_bits) - 1)) + 1 : signed_delta & ((1 << frac_bits) - 1);

    uint64_t wid_frac_BXdel = wid_B + wid_delta;
    uint64_t wid_frac_CXdel = wid_C + wid_delta + wid_delta;

    // 4. 定义各项的基础符号 (SQRT 展开: A正, B正, C负)
    uint64_t sign_A = 0;
    uint64_t sign_B = 0;
    uint64_t sign_C = 1; // 标记 1 代表在 fix_multi 里做减法

    // B 项的最终增减取决于 delta 的正负
    uint64_t final_sign_B = sign_delta ^ sign_B;

    // 5. 确定最大的对齐位宽
    uint64_t max_width = max(max(wid_frac_BXdel, wid_frac_CXdel), (uint64_t)wid_A);

    return fix_multi((uint64_t)A, B, C, ABS_delta,
                     max_width - wid_A,
                     max_width - wid_frac_BXdel,
                     max_width - wid_frac_CXdel,
                     sign_A, final_sign_B, sign_C);
}

static uint64_t LOG2_fix_multi(uint32_t A, uint32_t B, uint32_t C, uint32_t delta,
                               uint32_t wid_A, uint32_t wid_B, uint32_t wid_C, uint32_t table_idx,
                               uint32_t wid_delta = FP32_MANT_WIDTH)
{
    // 1. 动态计算当前区间的中心点偏移
    // 例如：128段表 (7 bit)，剩余 16 bit，中心点就在 1 << 15
    uint32_t frac_bits = FP32_MANT_WIDTH - LOG2_TABLE_BIT_WIDTH;
    uint32_t center_offset = 1 << (frac_bits - 1);

    // 2. 以中心点为 0，计算有符号的 delta
    int32_t signed_delta = (int32_t)delta - center_offset;

    // 3. 提取符号和绝对值
    uint64_t sign_delta = (signed_delta < 0) ? 1 : 0;
    uint64_t ABS_delta = sign_delta ? (~signed_delta & ((1 << frac_bits) - 1)) + 1 : signed_delta & ((1 << frac_bits) - 1);

    uint64_t wid_frac_BXdel = wid_B + wid_delta;
    uint64_t wid_frac_CXdel = wid_C + wid_delta + wid_delta;

    // 4. 定义各项的基础符号 (SQRT 展开: A正, B正, C负)
    uint64_t sign_A = 0;
    uint64_t sign_B = 0;
    uint64_t sign_C = 1; // 标记 1 代表在 fix_multi 里做减法

    // B 项的最终增减取决于 delta 的正负
    uint64_t final_sign_B = sign_delta ^ sign_B;

    // 5. 确定最大的对齐位宽
    uint64_t max_width = max(max(wid_frac_BXdel, wid_frac_CXdel), (uint64_t)wid_A);

    return fix_multi((uint64_t)A, B, C, ABS_delta,
                     max_width - wid_A,
                     max_width - wid_frac_BXdel,
                     max_width - wid_frac_CXdel,
                     sign_A, final_sign_B, sign_C);
}

// 修正后的 SQRT_fix_multi，彻底告别硬编码
static uint64_t RSQRT_fix_multi(uint32_t A, uint32_t B, uint32_t C, uint32_t delta,
                                uint32_t wid_A, uint32_t wid_B, uint32_t wid_C, uint32_t table_idx,
                                uint32_t wid_delta = FP32_MANT_WIDTH)
{
    // 1. 动态计算当前区间的中心点偏移
    // 例如：128段表 (7 bit)，剩余 16 bit，中心点就在 1 << 15
    uint32_t frac_bits = FP32_MANT_WIDTH - SQRT_TABLE_BIT_WIDTH;
    uint32_t center_offset = 1 << (frac_bits - 1);

    // 2. 以中心点为 0，计算有符号的 delta
    int32_t signed_delta = (int32_t)delta - center_offset;

    // 3. 提取符号和绝对值
    uint64_t sign_delta = (signed_delta < 0) ? 1 : 0;
    uint64_t ABS_delta = sign_delta ? (~signed_delta & ((1 << frac_bits) - 1)) + 1 : signed_delta & ((1 << frac_bits) - 1);

    uint64_t wid_frac_BXdel = wid_B + wid_delta;
    uint64_t wid_frac_CXdel = wid_C + wid_delta + wid_delta;

    // 4. 定义各项的基础符号 (SQRT 展开: A正, B正, C负)
    uint64_t sign_A = 0;
    uint64_t sign_B = 1;
    uint64_t sign_C = 0; // 标记 1 代表在 fix_multi 里做减法

    // B 项的最终增减取决于 delta 的正负
    uint64_t final_sign_B = sign_delta ^ sign_B;

    // 5. 确定最大的对齐位宽
    uint64_t max_width = max(max(wid_frac_BXdel, wid_frac_CXdel), (uint64_t)wid_A);

    return fix_multi((uint64_t)A, B, C, ABS_delta,
                     max_width - wid_A,
                     max_width - wid_frac_BXdel,
                     max_width - wid_frac_CXdel,
                     sign_A, final_sign_B, sign_C);
}

static uint64_t RCP_fix_multi(uint32_t A, uint32_t B, uint32_t C, uint32_t delta,
                              uint32_t wid_A, uint32_t wid_B, uint32_t wid_C,
                              uint32_t wid_delta = FP32_MANT_WIDTH) // a + bx + cx^2
{
    // 13(A) 7(B) 3(C) 11(delta, it's mantissa[5:0])
    // A/B/C are all without sign. Actually A & C are with sign 0, while B is with sign 1
    // delta is mantissa[14:0] including a sign as the reference point is in the middle of the section
    // of 1/256
    uint32_t sign_delta_pos = FP32_MANT_WIDTH - RCP_TABLE_BIT_WIDTH - 1;                                                // compute delta width
    uint64_t sign_delta = (delta >> sign_delta_pos) & 0x1 ? 0 : 1;                                                      // get delta sign
    uint64_t ABS_delta = sign_delta ? (~delta & ((1 << sign_delta_pos) - 1)) + 1 : delta & ((1 << sign_delta_pos) - 1); // non sign delta

    uint64_t wid_B_mul_del = wid_B + wid_delta;                             // bx width
    uint64_t wid_C_mul_del_del = wid_C + wid_delta + wid_delta;             // cx^2 width
    uint64_t max_width = max(max(wid_B_mul_del, wid_C_mul_del_del), wid_A); // max width
    uint64_t sign_B = 1;
    uint64_t sign_C = 0;
    return fix_multi((uint64_t)A, B, C, ABS_delta, max_width - wid_A, max_width - wid_B_mul_del, 0, 0, sign_delta ^ sign_B, sign_C);

    // uint64_t B_mul_del = ABS_delta * B;                 // 0.23 * 0.15 -> 0.30
    // uint64_t C_mul_del_del = ABS_delta * ABS_delta * C; // 0.23 * 0.23 * 0.8 -> 0.54

    // // align to the same format as A : 0.13
    // uint64_t tempA = (uint64_t)A << (max_width - wid_A);
    // B_mul_del = (B_mul_del << ());

    // uint32_t sign_B_mul_del = ;
    // uint32_t sign_C_mul_del_del = sign_C;

    // uint64_t res;
    // B_mul_del = sign_B_mul_del ? ~B_mul_del + 1 : B_mul_del;
    // C_mul_del_del = sign_C_mul_del_del ? ~C_mul_del_del + 1 : C_mul_del_del;

    // res = tempA + B_mul_del + C_mul_del_del;

    // return res;
}

// static uint64_t SIN_fix_multi(uint64_t A, uint32_t B, uint32_t C, uint32_t delta,
//                               uint32_t wid_A, uint32_t wid_B, uint32_t wid_C, uint32_t exp,
//                               uint32_t wid_delta, uint32_t valid_bits)
// {
//     // 1. 动态计算当前区间的中心点偏移
//     // 依据 valid_bits (如 20)，中心点刚好在 1 << 19 处
//     uint32_t center_offset = 1 << (valid_bits - 1);

//     // 2. 以中心点为 0，计算有符号的 delta
//     int32_t signed_delta = (int32_t)delta - center_offset;

//     // 3. 提取符号和绝对值 (修复了三元运算符 Bug)
//     uint64_t sign_delta = (signed_delta < 0) ? 1 : 0;
//     // 取绝对值：负数取反加一，正数保留有效位
//     uint64_t ABS_delta = sign_delta ? ((~signed_delta & ((1 << valid_bits) - 1)) + 1)
//                                     : (signed_delta & ((1 << valid_bits) - 1));

//     // 4. 计算 SIN 特有的动态指数补偿 (moreShift)
//     uint32_t tempShift = valid_bits - 17 - 1;   // 面积优化：截断送入 ALU 的 delta 精度
//     uint32_t moreShift = 127 - exp - tempShift; // 浮点阶码带来的额外缩放

//     // 执行截断
//     ABS_delta >>= tempShift;

//     // 5. 将原本繁琐的乘法后右移，等效转化为“目标小数位宽的增加”
//     // 数学原理：在定点数中，结果右移 moreShift 等价于它的小数位宽增加了 moreShift
//     uint64_t wid_frac_BXdel = (int32_t)wid_B - 3 + wid_delta;
//     uint64_t wid_frac_CXdel = (int32_t)wid_C - 5 + wid_delta + wid_delta;

//     // 6. 定义各项的基础符号 (SIN 泰勒/Minimax 展开: A正, B看delta, C为二阶导恒负)
//     uint64_t sign_A = 0;
//     uint64_t sign_B = 0;
//     uint64_t sign_C = 1; // 标记 1 代表在 fix_multi 里做减法

//     // B 项的最终增减取决于 delta 的正负
//     uint64_t final_sign_B = sign_delta ^ sign_B;

//     // 7. 确定最大的对齐位宽
//     uint64_t max_width = max(max(wid_frac_BXdel, wid_frac_CXdel), (uint64_t)wid_A);
//     assert(max_width < 60);
//     // 8. 调用底层的通用乘加树引擎
//     // 计算出的 max_width 减去各自的 wid_frac 即为各自需要补齐的左移位数 (shift_w)
//     return fix_multi(A, B, C, ABS_delta,
//                      max_width - wid_A,
//                      max_width - wid_frac_BXdel,
//                      max_width - wid_frac_CXdel,
//                      sign_A, final_sign_B, sign_C, moreShift);
// }

static uint64_t SIN_fix_multi(uint64_t A, uint32_t B, uint32_t C, uint32_t delta,
                              uint32_t wid_A, uint32_t wid_B, uint32_t wid_C, uint32_t exp,
                              uint32_t wid_delta, uint32_t valid_bits)
{
    uint32_t center_offset = 1 << (valid_bits - 1);
    int32_t signed_delta = (int32_t)delta - center_offset;
    uint64_t sign_delta = (signed_delta < 0) ? 1 : 0;

    uint64_t ABS_delta = sign_delta ? ((~signed_delta & ((1 << valid_bits) - 1)) + 1)
                                    : (signed_delta & ((1 << valid_bits) - 1));

    // 使用 int32_t 彻底防止大角度无符号下溢
    int32_t tempShift = (int32_t)valid_bits - 18;
    int32_t moreShift = 127 - (int32_t)exp - tempShift;

    if (tempShift > 0)
        ABS_delta >>= tempShift;
    else if (tempShift < 0)
        ABS_delta <<= (-tempShift);

    // 💡 核心：基础位宽绝对不包含 moreShift，保证 max_width 恒定！
    uint64_t wid_frac_BXdel = (int32_t)wid_B - 3 + wid_delta;
    uint64_t wid_frac_CXdel = (int32_t)wid_C - 5 + wid_delta + wid_delta;

    uint64_t sign_A = 0;
    uint64_t sign_B = 0;
    uint64_t sign_C = 1;

    uint64_t final_sign_B = sign_delta ^ sign_B;

    // 算出来的 max_width 将会完美定死在 51 (若A=27, B=18, C=10)
    uint64_t max_width = max(max(wid_frac_BXdel, wid_frac_CXdel), (uint64_t)wid_A);
    // 你的硬件断言
    // assert(max_width < 60);

    return fix_multi(A, B, C, ABS_delta,
                     max_width - wid_A,
                     max_width - wid_frac_BXdel,
                     max_width - wid_frac_CXdel,
                     sign_A, final_sign_B, sign_C, (uint32_t)moreShift);
}

uint32_t fp32_rcp(uint32_t src);
uint32_t fp32_sqrt(uint32_t src);
uint32_t fp32_rsq(uint32_t src);
uint32_t fp32_log2(uint32_t src);
uint32_t fp32_sin(uint32_t src, bool ftz);
uint32_t fp32_cos(uint32_t src, bool ftz);
