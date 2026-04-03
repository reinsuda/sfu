#include "utils.h"
#include "log2_coeffs.h"

#include "iostream"

uint32_t fp32_log2(uint32_t src)
{

    Precision pre;
    pre.A_pre = 27;
    pre.B_pre = 18;
    pre.C_pre = 10;
    // special number handle
    uint32_t sign = src & 0x80000000;
    uint32_t nonsign = sign ^ src;

    if (fp32_is_zero(src))
    {
        return 0x80000000 | FP32_INF;
    }
    if (fp32_is_nan(src) || sign)
    {
        return 0xFFFFFFFF;
    }
    if (fp32_is_inf(src))
    {
        return FP32_INF;
    }
    int32_t exp = (nonsign >> FP32_MANT_WIDTH) & N_BIT_1(FP32_EXP_WIDTH);
    uint32_t mant = nonsign & N_BIT_1(FP32_MANT_WIDTH);
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

    uint32_t lut_id = (mant >> (FP32_MANT_WIDTH - LOG2_TABLE_BIT_WIDTH)) & N_BIT_1(LOG2_TABLE_BIT_WIDTH);
    uint32_t delta = mant & (N_BIT_1(FP32_MANT_WIDTH - LOG2_TABLE_BIT_WIDTH));
    rcp_entry_t table = FP32_LOG2_TABLE[lut_id];
    exp -= 127;

    uint32_t c2_abs = table.c2 & ((1 << pre.C_pre) - 1);     // 提取绝对值 32
    uint32_t c1_abs = table.c1_abs & ((1 << pre.B_pre) - 1); // 提取绝对值 32
    uint32_t c0_abs = table.c0 & ((1 << pre.A_pre) - 1);     // 提取绝对值 32
    uint64_t table_res = LOG2_fix_multi(table.c0, table.c1_abs, c2_abs, delta, pre.A_pre, pre.B_pre, pre.C_pre, 1);

    // conver to 0.23
    uint64_t wid_frac_BXdel = pre.B_pre + FP32_MANT_WIDTH;
    uint64_t wid_frac_CXdel = pre.C_pre + (2 * FP32_MANT_WIDTH);
    uint64_t decimal_bits = max(max(wid_frac_BXdel, wid_frac_CXdel), (uint64_t)pre.A_pre);

    uint64_t resFix;
    uint32_t resSign;

    // 💡 核心修复：为了防止 exp<<decimal_bits 溢出到第 63 位被当成负数
    // 我们强制腾出 2 个 bit 的安全区
    uint64_t safe_decimal_bits = decimal_bits - 2;

    // 把定点小数也同步右移 2 位对齐
    // (由于有高达 56 位的定点小数，丢掉最底部的 2 位对 FP32 的精度毫无影响)
    uint64_t safe_table_res = table_res >> 2;

    if (exp < 0)
    {
        resFix = ((uint64_t)abs(exp) << safe_decimal_bits) - safe_table_res;
        resSign = 1;
    }
    else
    {
        resFix = ((uint64_t)exp << safe_decimal_bits) + safe_table_res;
        resSign = 0;
    }

    // 完美传入绝对值 resFix，注意传入的是 safe_decimal_bits!
    uint32_t test_rst = NormalizeToFP32(resFix, 0x7f, safe_decimal_bits);

    // 组装最终结果
    return test_rst | (resSign << 31);
}