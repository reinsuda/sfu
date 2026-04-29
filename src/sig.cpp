#include "utils.h"
#include "sig_coeffs.h"
#include "exp2_coeffs.h"

uint64_t SIG_fix_multi(uint32_t A, uint32_t B, uint32_t C, uint32_t delta,
                       uint32_t wid_A, uint32_t wid_B, uint32_t wid_C,
                       uint32_t delta_bits, uint32_t isExp, uint32_t wid_delta = FP32_MANT_WIDTH) // a + bx + cx^2
{
    // 1. 解析 delta 的符号位与绝对值
    // delta is with a sign on [sign_pos]
    uint32_t sign_pos = delta_bits - 1;
    uint32_t sign_delta = (delta >> sign_pos) & 0x1;
    uint64_t ABS_delta = !sign_delta ? (~delta & ((1 << sign_pos) - 1)) + 1 : delta & ((1 << sign_pos) - 1);

    if (isExp)
    {
        sign_delta = !sign_delta;
    }

    // 2. 计算各项的小数位宽 (Fractional width)
    uint64_t wid_frac_A = wid_A - isExp;                     // 如果 isExp=1, A 含有 1 个整数位
    uint64_t wid_frac_BXdel = wid_B - isExp + wid_delta;     // B*delta 的小数位宽
    uint64_t wid_frac_CXdel = wid_C + wid_delta + wid_delta; // C*delta^2 的小数位宽

    // 3. 找到最大的小数位宽，将所有结果对齐到该小数点
    uint64_t max_width = std::max(std::max(wid_frac_BXdel, wid_frac_CXdel), wid_frac_A);

    // 4. 计算为了对齐目标小数点需要进行的左移位数
    uint32_t A_shift_w = max_width - wid_frac_A;
    uint32_t B_shift_w = max_width - wid_frac_BXdel;
    uint32_t c_shift_w = max_width - wid_frac_CXdel;

    // 5. 确定最终项的符号
    // 原代码中 sign_A=0, sign_B=0, sign_C=0
    uint32_t a_sign = 0;
    uint32_t b_sign = sign_delta ^ 1; // 继承 delta 和 B 的异或结果
    uint32_t c_sign = 0;

    // 6. 调用底层定点数乘加树 (统一交由 fix_multi 处理移位与补码逻辑)
    return fix_multi((uint64_t)A, B, C, ABS_delta,
                     A_shift_w, B_shift_w, c_shift_w,
                     a_sign, b_sign, c_sign, 0);
}

uint32_t getSigTableId(const uint32_t exp, const uint32_t mant, uint32_t &delta, uint32_t &delta_bits)
{
    uint32_t t_idx = 0;

    switch (exp)
    {
    case 130:
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = 160 + (mant >> delta_bits); // 5-bit table, so need the upper 6-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    case 129:
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = 128 + (mant >> delta_bits); // 6-bit table, so need the upper 6-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    case 128:
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = 96 + (mant >> delta_bits); // 5-bit table, so need the upper 4-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    case 127: // exp == 0
        delta_bits = FP32_MANT_WIDTH - 5;
        t_idx = 64 + (mant >> delta_bits); // 4-bit table, so need the upper 4-bit of the mantissa
        delta = mant & N_BIT_1(delta_bits);
        break;
    default: // exp < 127
        uint32_t fix_value = ((1 << FP32_MANT_WIDTH) | mant) >> (127 - exp);
        delta_bits = FP32_MANT_WIDTH - 6;
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
        if (sign != 0)
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