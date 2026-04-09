#include "utils.h"
#include "sig_coeffs.h"
#include "exp2_coeffs.h"

uint32_t getSigTableId(const uint32_t exp, const uint32_t mant, uint32_t &delta, uint32_t &delta_bits)
{
    uint32_t t_idx = 0;

    switch (exp)
    {
    case 130:
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = 32 + 16 + 32 + 64 + (mant >> delta_bits); // 5-bit table, so need the upper 6-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    case 129:
        delta_bits = FP32_MANT_WIDTH - 6;
        t_idx = 32 + 16 + 32 + (mant >> delta_bits); // 6-bit table, so need the upper 6-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    case 128:
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = 32 + 16 + (mant >> delta_bits); // 5-bit table, so need the upper 4-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    case 127: // exp == 0
        delta_bits = FP32_MANT_WIDTH - 4;
        t_idx = 32 + (mant >> delta_bits); // 4-bit table, so need the upper 4-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    default: // exp < 127
        uint32_t fix_value = ((1 << FP32_MANT_WIDTH) | mant) >> (127 - exp);
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = fix_value >> delta_bits; // 5-bit table, so need the upper 5-bit of the mantissa
        delta = fix_value & N_BIT_1(delta_bits);
        break;
    }

    return t_idx;
}

void cvtToFix(uint32_t exp, uint32_t mant, int32_t &newExp, uint32_t &decimal, bool &isINF)
{
    // convert the floating point to (M + 0.N)
    int32_t expNoBias = exp - 127;
    uint32_t fixPoint = 1 << FP32_MANT_WIDTH | mant;
    if (expNoBias >= 0)
    {
        if (expNoBias >= 8) // overflow on 8bits exp, set to INF value
            isINF = true;
        else
        {
            newExp = ((fixPoint << expNoBias) >> FP32_MANT_WIDTH) & 0xff;
            decimal = (fixPoint << expNoBias) & N_BIT_1(FP32_MANT_WIDTH);
            if (newExp >= 128) // as 128+127 >=255 which is an INF number
                isINF = true;
        }
    }
    else if (expNoBias <= -32)
        decimal = 0;
    else
        decimal = (fixPoint >> -expNoBias) & N_BIT_1(FP32_MANT_WIDTH);

    // for sigmoid, we always use negative number
    newExp = decimal != 0 ? -newExp - 1 : -newExp;
    decimal = decimal != 0 ? ((~decimal) & N_BIT_1(FP32_MANT_WIDTH)) + 1 : 0;
}

uint32_t fp32_sig(uint32_t src)
{

    Precision pre;
    pre.A_pre = 27;
    pre.B_pre = 18;
    pre.C_pre = 13;
    // special number handle
    uint32_t sign = src & 0x80000000;
    uint32_t nonsign = sign ^ src;
    if (fp32_is_nan(src))
    {
        return 0xFFFFFFFF;
    }
    if (fp32_is_inf(src))
    {
        if (sign)
        {
            return 0;
        }
        else
        {
            return 0x3f800000;
        }
    }
    if (nonsign >= 0x41380000)
    {
        if (sign)
        {
            return 0;
        }
        else
        {
            return 0x3f800000;
        }
    }
    int32_t exp = (nonsign >> FP32_MANT_WIDTH) & N_BIT_1(FP32_EXP_WIDTH);
    uint32_t mant = nonsign & N_BIT_1(FP32_MANT_WIDTH);
    int32_t expNoBias = (int32_t)exp - 127;
    int32_t newExp = 0;
    bool isINF = false;
    if (exp == 0) // denorm handle
    {
        uint32_t lz = __builtin_clz(mant);

        // 正确的左移位数 (应为正数 1 或 2)
        uint32_t shift_amt = lz - 8;

        // 指数补偿：起始等效阶码为1，每左移一位，阶码减一
        exp = 9 - lz; // 等价于 exp = 1 - shift_amt;

        // 尾数左移，并掩去已经跑到第 23 位的那个隐含的 '1'
        mant = (mant << shift_amt) & N_BIT_1(FP32_MANT_WIDTH);
    }
    uint32_t lut_id = 0;
    uint32_t delta = 0;
    uint32_t delta_bit = 0;
    bool is_exp = false;
    rcp_entry_t table;
    uint64_t table_res;
    if (exp <= 130)
    {
        lut_id = getSigTableId(exp, mant, delta, delta_bit);
        table = FP32_SIG_TABLE[lut_id];
    }
    else // use exp table exp >=130
    {
        is_exp = true;
        uint32_t decimal = 0;
        cvtToFix(exp, mant, newExp, decimal, isINF);
        delta_bit = FP32_MANT_WIDTH - EXP_TABLE_BIT_WIDTH;
        lut_id = (decimal >> delta_bit) & N_BIT_1(EXP_TABLE_BIT_WIDTH);
        delta = decimal & N_BIT_1(delta_bit);
        table = FP32_EXP2_TABLE[lut_id];
    }
    table_res = SIG_fix_multi(table.c0, table.c1_abs, table.c2, delta, pre.A_pre, pre.B_pre, pre.C_pre, delta_bit, is_exp);
    if (exp < 116)
    {
        return 0x3f000000;
    }
    uint32_t wid_frac_A = pre.A_pre - is_exp;
    uint32_t wid_frac_BXdel = pre.B_pre - is_exp + FP32_MANT_WIDTH;
    uint32_t wid_frac_CXdel = pre.C_pre + FP32_MANT_WIDTH * 2;
    uint32_t max_width = std::max(std::max(wid_frac_BXdel, wid_frac_CXdel), wid_frac_A);

    uint32_t rst = 0;

    // 💡 严格解耦后处理逻辑
    if (!is_exp)
    {
        // =====================================
        // 路径 A: 走的是 SIG_TABLE (结果在 0.5~1 之间)
        // =====================================
        if (sign == 0)
        {
            // 正数输入：计算 1.0 - table_res
            // (1ULL << max_width) 完美代表定点数的 1.0，不会有任何位宽截断！
            table_res = (1ULL << max_width) - table_res;
        }

        // SIG 表的结果永远视为基准阶码 127
        rst = table_res != 0 ? NormalizeToFP32(table_res, 127, max_width) : 0;
    }
    else
    {
        // =====================================
        // 路径 B: 走的是 EXP_TABLE (处理 |x| >= 8.0 的长尾)
        // =====================================
        if (sign == 0)
        {
            // 正数输入：计算 1.0 - e^{-x}
            // 因为走的是 EXP 表，e^{-x} 必须根据 newExp 右移对齐
            uint64_t shifted_res = (-newExp >= 60) ? 0 : (table_res >> -newExp);
            table_res = (1ULL << max_width) - shifted_res;

            rst = table_res != 0 ? NormalizeToFP32(table_res, 127, max_width) : 0;
        }
        else
        {
            // 负数输入：直接计算 e^x
            rst = table_res != 0 ? NormalizeToFP32(table_res, newExp + 127, max_width) : 0;
        }
    }

    return rst;
}