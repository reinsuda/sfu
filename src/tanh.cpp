#include "utils.h"
#include "tanh_coeffs.h"

// uint64_t tan_fix_multi(uint32_t A, uint32_t B, uint32_t C, uint32_t delta,
//                        uint32_t wid_A, uint32_t wid_B, uint32_t wid_C,
//                        uint32_t delta_bits, uint32_t wid_delta = FP32_MANT_WIDTH) // a + bx + cx^2
// {
//     // 1. 解析 delta 的符号位与绝对值
//     // delta is with a sign on [sign_pos]
//     uint32_t sign_pos = delta_bits - 1;
//     uint32_t sign_delta = (delta >> sign_pos) & 0x1;
//     uint64_t ABS_delta = !sign_delta ? (~delta & ((1 << sign_pos) - 1)) + 1 : delta & ((1 << sign_pos) - 1);

//     // 2. 计算各项的小数位宽 (Fractional width)
//     uint64_t wid_frac_A = wid_A;                             // 如果 isExp=1, A 含有 1 个整数位
//     uint64_t wid_frac_BXdel = wid_B + wid_delta;             // B*delta 的小数位宽
//     uint64_t wid_frac_CXdel = wid_C + wid_delta + wid_delta; // C*delta^2 的小数位宽

//     // 3. 找到最大的小数位宽，将所有结果对齐到该小数点
//     uint64_t max_width = std::max(std::max(wid_frac_BXdel, wid_frac_CXdel), wid_frac_A);

//     // 4. 计算为了对齐目标小数点需要进行的左移位数
//     uint32_t A_shift_w = max_width - wid_frac_A;
//     uint32_t B_shift_w = max_width - wid_frac_BXdel;
//     uint32_t c_shift_w = max_width - wid_frac_CXdel;

//     // 5. 确定最终项的符号
//     // 原代码中 sign_A=0, sign_B=0, sign_C=0
//     uint32_t a_sign = 0;
//     uint32_t b_sign = sign_delta ^ 0; // 继承 delta 和 B 的异或结果
//     uint32_t c_sign = 1;

//     // 6. 调用底层定点数乘加树 (统一交由 fix_multi 处理移位与补码逻辑)
//     return fix_multi((uint64_t)A, B, C, ABS_delta,
//                      A_shift_w, B_shift_w, c_shift_w,
//                      a_sign, b_sign, c_sign, 0);
// }

uint32_t getTanhTableId(const uint32_t exp, const uint32_t mant, uint32_t &delta, uint32_t &delta_bits)
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
uint64_t tan_fix_multi(uint32_t A, uint32_t B, uint32_t C, uint32_t delta,
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

uint32_t fp32_tanh(uint32_t src)
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
        return sign | 0x3f800000;
    if (fp32_is_zero(src))
        return src;

    int32_t exp = (nonsign >> FP32_MANT_WIDTH) & N_BIT_1(FP32_EXP_WIDTH);
    uint32_t mant = nonsign & N_BIT_1(FP32_MANT_WIDTH);

    if (exp <= 119)
        return src; // 👈 阈值必须是 119

    uint32_t lut_id = 0, delta = 0, delta_bit = 0;
    rcp_entry_t table;

    if (exp <= 130)
    {
        lut_id = getTanhTableId(exp, mant, delta, delta_bit);
        table = FP32_TANH_TABLE[lut_id];
    }
    else
    {
        return sign | 0x3f800000;
    }

    if (exp == 130)
    {
        uint32_t rst = NormalizeToFP32(table.c0, 127, pre.A_pre);
        return sign | rst;
    }

    // 💡 注意看这里：正好传了 7 个参数！跟上面完美匹配！
    uint64_t table_res = tan_fix_multi(table.c0, table.c1_abs, table.c2, delta,
                                       pre.A_pre, pre.B_pre, pre.C_pre);

    // 💡 外层对齐也写死为 23
    uint32_t wid_frac_BXdel = pre.B_pre + 23;
    uint32_t wid_frac_CXdel = pre.C_pre + 46;
    uint32_t max_width = std::max(std::max(wid_frac_BXdel, wid_frac_CXdel), pre.A_pre);

    uint32_t rst = table_res != 0 ? NormalizeToFP32(table_res, 127, max_width) : 0;

    return sign | rst;
}
