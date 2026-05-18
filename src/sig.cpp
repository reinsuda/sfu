#include "utils.h"
#include "tanh_coeffs.h"
#include "exp2_coeffs.h"
#include "sigmoid_neg_tail.h"

uint32_t getSigTableId(const uint32_t exp, const uint32_t mant, uint32_t &delta, uint32_t &delta_bits)
{
    uint32_t t_idx = 0;

    switch (exp)
    {
    case 130: // 8~16
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = 128 + (mant >> delta_bits); // 5-bit table, so need the upper 6-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    case 129: // 4~8
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = 96 + (mant >> delta_bits); // 6-bit table, so need the upper 6-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    case 128: // 2~4
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = 64 + (mant >> delta_bits); // 5-bit table, so need the upper 4-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    case 127: // exp == 0  1~2
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = 32 + (mant >> delta_bits); // 4-bit table, so need the upper 4-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    default: // exp < 127  0~1
        uint32_t fix_value = ((1 << FP32_MANT_WIDTH) | mant) >> (127 - exp);
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = fix_value >> delta_bits; // 5-bit table, so need the upper 5-bit of the mantissa
        delta = fix_value & N_BIT_1(delta_bits);
        break;
    }

    return t_idx;
}

// 💡 纯净版乘加树：只有 7 个参数，没有任何符号判断，纯 64 位防溢出！
uint64_t sig_fix_multi(uint32_t A, uint32_t B, uint32_t C, uint32_t delta,
                       uint32_t wid_A, uint32_t wid_B, uint32_t wid_C)
{
    uint32_t wid_delta = 23;  // delta 小数位权永远是 23
    uint64_t u_delta = delta; // 纯绝对值，不作任何修改！

    uint64_t wid_frac_A = wid_A;
    uint64_t wid_frac_BXdel = wid_B + wid_delta;
    uint64_t wid_frac_CXdel = wid_C + wid_delta + wid_delta;

    uint64_t max_width = std::max(std::max(wid_frac_BXdel, wid_frac_CXdel), wid_frac_A);

    uint32_t A_shift_w = max_width - wid_frac_A;
    uint32_t B_shift_w = max_width - wid_frac_BXdel;
    uint32_t c_shift_w = max_width - wid_frac_CXdel;

    uint64_t termA = (uint64_t)A << A_shift_w;
    uint64_t termB = ((uint64_t)B * u_delta) << B_shift_w;
    uint64_t termC = ((uint64_t)C * u_delta * u_delta) << c_shift_w;

    // A + Bx - Cx^2
    return (termA + termB > termC) ? (termA + termB - termC) : 0;
}

// 💡 纯净版乘加树：只有 7 个参数，没有任何符号判断，纯 64 位防溢出！
uint64_t sig_fix_multi_neg(uint32_t A, uint32_t B, uint32_t C, uint32_t delta,
                           uint32_t wid_A, uint32_t wid_B, uint32_t wid_C)
{
    uint32_t wid_delta = 23;  // delta 小数位权永远是 23
    uint64_t u_delta = delta; // 纯绝对值，不作任何修改！

    uint64_t wid_frac_A = wid_A;
    uint64_t wid_frac_BXdel = wid_B + wid_delta;
    uint64_t wid_frac_CXdel = wid_C + wid_delta + wid_delta;

    uint64_t max_width = std::max(std::max(wid_frac_BXdel, wid_frac_CXdel), wid_frac_A);

    uint32_t A_shift_w = max_width - wid_frac_A;
    uint32_t B_shift_w = max_width - wid_frac_BXdel;
    uint32_t c_shift_w = max_width - wid_frac_CXdel;

    uint64_t termA = (uint64_t)A << A_shift_w;
    uint64_t termB = ((uint64_t)B * u_delta) << B_shift_w;
    uint64_t termC = ((uint64_t)C * u_delta * u_delta) << c_shift_w;

    // A + Bx - Cx^2
    return (termA + termC - termB);
}

uint32_t fp32_sig(uint32_t src)
{
    Precision pre;
    pre.A_pre = 27;
    pre.B_pre = 17; // 👈 必须是 17
    pre.C_pre = 13;

    uint32_t sign = src & 0x80000000;
    uint32_t nonsign = sign ^ src;

    if (fp32_is_nan(src))
        return 0xFFFFFFFF;
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

    if (fp32_is_zero(src))
        return 0x3f000000;

    int32_t exp = sign && ((nonsign >> FP32_MANT_WIDTH) & N_BIT_1(FP32_EXP_WIDTH)) > 129 ? ((nonsign >> FP32_MANT_WIDTH) & N_BIT_1(FP32_EXP_WIDTH)) : ((nonsign >> FP32_MANT_WIDTH) & N_BIT_1(FP32_EXP_WIDTH)) - 1;
    uint32_t mant = nonsign & N_BIT_1(FP32_MANT_WIDTH);

    if (exp <= 112)
    {
        return 0x3f000000; // 👈 阈值必须是 119
    }

    uint32_t lut_id = 0, delta = 0, delta_bit = 0;
    rcp_entry_t table;
    // if (src > 0xc1100000)
    // {
    //     return 0;
    // }
    if (exp <= 130)
    {
        lut_id = getSigTableId(exp, mant, delta, delta_bit);
        table = FP32_TANH_TABLE[lut_id];
    }
    else if (!sign)
    {
        return sign ? 0 : 0x3f800000; // 返回 1.0
    }

    if (exp == 130 && !sign)
    {
        if (lut_id == 0x80)
        {
            return NormalizeToFP32(0x0fffffe8, 126, pre.A_pre);
        }
        else if (lut_id == 0x81)
        {
            return NormalizeToFP32(0x0ffffff1, 126, pre.A_pre);
        }
        else if (lut_id == 0x82)
        {
            return NormalizeToFP32(0x0ffffff7, 126, pre.A_pre);
        }
        else if (lut_id == 0x83)
        {
            return NormalizeToFP32(0x0ffffffb, 126, pre.A_pre);
        }
        else if (lut_id == 0x84)
        {
            return NormalizeToFP32(0x0ffffffd, 126, pre.A_pre);
        }
        else if (lut_id == 0x85)
        {
            return NormalizeToFP32(0x0ffffffe, 126, pre.A_pre);
        }
        else if (lut_id == 0x86 || lut_id == 0x87)
        {
            return NormalizeToFP32(0x0fffffff, 126, pre.A_pre);
        }
        else
        {
            // std::cout << std::hex << "src: " << src << std::endl;
            return 0x3f800000; // 返回 1.0
        }
    }
    uint32_t wid_frac_BXdel = pre.B_pre + 23;
    uint32_t wid_frac_CXdel = pre.C_pre + 46;
    uint32_t max_width = std::max(std::max(wid_frac_BXdel, wid_frac_CXdel), pre.A_pre);
    if (sign && exp > 129)
    {
        if (src >= 0xc2b17218)
        {
            return 0;
        }
        uint64_t tm = (1 << FP32_MANT_WIDTH) | mant;
        uint64_t tm1 = (uint64_t)tm * (uint64_t)0xB8AA3B;
        tm1 >>= FP32_MANT_WIDTH;
        uint32_t expt = exp;
        if (((tm1 >> (FP32_MANT_WIDTH + 1)) & 0x1) != 0)
        {
            expt++;
            tm1 >>= 1;
        }
        uint32_t new_src = 0x80000000 | (expt << FP32_MANT_WIDTH) | (tm1 & 0x7fffff);
        return fp32_exp2(new_src);
    }
    // 💡 注意看这里：正好传了 7 个参数！跟上面完美匹配！
    uint64_t table_res = sig_fix_multi(table.c0, table.c1_abs, table.c2, delta,
                                       pre.A_pre, pre.B_pre, pre.C_pre);

    // 💡 外层对齐也写死为 23
    if (!sign)
        table_res += (uint64_t)1 << max_width;
    else
    {
        table_res = ((uint64_t)1 << max_width) - table_res;
    }
    uint32_t rst = table_res != 0 ? NormalizeToFP32(table_res, 126, max_width) : 0;

    return rst;
}
