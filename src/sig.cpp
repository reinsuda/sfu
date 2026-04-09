#include "utils.h"
#include "sig_coeffs.h"
#include "exp2_coeffs.h"

uint32_t getSigTableId(const uint32_t exp, const uint32_t mant, uint32_t &delta, uint32_t &delta_bits)
{
    uint32_t t_idx = 0;

    switch (exp)
    {
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
        t_idx = 32 + (mant >> 7); // 4-bit table, so need the upper 4-bit of the mantissa
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

uint32_t fp32_sig(uint32_t src)
{

    Precision pre;
    pre.A_pre = 24;
    pre.B_pre = 18;
    pre.C_pre = 8;
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
    if (exp < 130)
    {
        lut_id = getSigTableId(exp, mant, delta, delta_bit);
        table = FP32_SIG_TABLE[lut_id];
    }
    else // use exp table
    {
    }
}