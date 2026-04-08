#include "utils.h"
#include "exp2_coeffs.h"
uint32_t fp32_exp2(uint32_t src)
{

    Precision pre;
    pre.A_pre = 25;
    pre.B_pre = 19;
    pre.C_pre = 13;

    // special number handle
    uint32_t sign = src & 0x80000000;
    uint32_t nonsign = sign ^ src;

    if (fp32_is_nan(src))
        return 0xFFFFFFFF;

    if (fp32_is_inf(src))
    {
        return sign ? 0 : 0x7F800000; // +INF返回INF，-INF返回0
    }

    // 提取 IEEE 754 字段
    int32_t exp = (nonsign >> 23) & 0xff;
    uint32_t mant = nonsign & 0x7fffff;

    // 💡 修复1：处理非规格化数 (Denormal) - 冲刷到 0 (Flush to Zero)
    if (exp == 0)
        mant = 0;

    // 💡 修复2：如果彻底是 0，结果就是 2^0 = 1.0
    if (exp == 0 && mant == 0)
        return 0x3f800000;

    uint64_t decimal = 0; // 0.N
    int32_t expNoBias = (int32_t)exp - 127;

    // 💡 修复3：如果是 0 或非规格化数，是没有隐含的 '1' 的！
    uint32_t sig = (exp == 0) ? 0 : ((1 << 23) | mant);

    int32_t newExp = 0; // M, will be used in normalization

    if (expNoBias >= 0)
    {
        if (expNoBias >= 8)                        // overflow on 8bits exp, set to INF value
            return sign ? 0x00000000 : 0x7F800000; // 💡 负数下溢给 0，正数溢出给 INF
        else
        {
            newExp = ((sig << expNoBias) >> 23) & 0xff;
            decimal = (sig << expNoBias) & 0x7fffff; // N_BIT_1(23)
            if (newExp >= 128)
                return sign ? 0x00000000 : 0x7F800000; // 💡 同上
        }
    }
    else if (expNoBias <= -32)
    {
        decimal = 0;
    }
    else
    {
        decimal = (sig >> -expNoBias) & 0x7fffff;
    }

    if (sign)
    {
        newExp = decimal != 0 ? -newExp - 1 : -newExp;
        decimal = decimal != 0 ? ((~decimal) & 0x7fffff) + 1 : 0;
    }

    // 💡 修复4 (核心修复)：查表索引(lut_id)和偏移量(delta) 必须从 decimal 提取！
    // decimal 是 23 位的：最高 7 位做索引 (对应 128 段)，剩余 16 位做 delta
    uint32_t lut_id = (decimal >> 16) & 0x7F;
    uint32_t delta = decimal & 0xFFFF;

    rcp_entry_t table = FP32_EXP2_TABLE[lut_id];

    // 调用定点数乘加树
    uint64_t table_res = EXP_fix_multi(table.c0, table.c1_abs, table.c2, delta, pre.A_pre, pre.B_pre, pre.C_pre);

    // 规格化：拼接阶码和尾数
    uint32_t test_rst = NormalizeToFP32(table_res, newExp + 127, pre.C_pre + 46); // 2 * FP32_MANT_WIDTH = 46

    return test_rst;
}