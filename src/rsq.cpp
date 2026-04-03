#include "utils.h"
#include "rsqrto_coeffs.h"
#include "rsqrte_coeffs.h"
#include "iostream"

uint32_t fp32_rsq(uint32_t src)
{

    Precision pre;
    pre.A_pre = 27;
    pre.B_pre = 18;
    pre.C_pre = 10;
    // special number handle
    uint32_t sign = src & 0x80000000;
    uint32_t nonsign = sign ^ src;
    if (fp32_is_nan(src) || sign)
    {
        return 0xFFFFFFFF;
    }
    if (fp32_is_zero(src))
    {
        return sign | FP32_INF;
    }
    if (fp32_is_inf(src))
    {
        return sign;
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

    uint32_t lut_id = (mant >> (FP32_MANT_WIDTH - SQRT_TABLE_BIT_WIDTH)) & N_BIT_1(SQRT_TABLE_BIT_WIDTH);
    uint32_t delta = mant & (N_BIT_1(FP32_MANT_WIDTH - SQRT_TABLE_BIT_WIDTH));
    rcp_entry_t table;
    exp -= 127;
    if (exp % 2 == 0)
    {
        table = FP32_RSQRT_E_TABLE[lut_id];
    }
    else
    {
        table = FP32_RSQRT_O_TABLE[lut_id];
    }
    exp = (exp < 0 && exp % 2 != 0) ? (exp / 2) - 1 : (exp / 2);
    uint32_t c2_abs = table.c2 & ((1 << pre.C_pre) - 1);     // 提取绝对值 32
    uint32_t c1_abs = table.c1_abs & ((1 << pre.B_pre) - 1); // 提取绝对值 32
    uint32_t c0_abs = table.c0 & ((1 << pre.A_pre) - 1);     // 提取绝对值 32
    uint64_t table_res = RSQRT_fix_multi(table.c0, table.c1_abs, c2_abs, delta, pre.A_pre, pre.B_pre, pre.C_pre, 1);

    // conver to 0.23
    uint64_t wid_frac_BXdel = pre.B_pre + FP32_MANT_WIDTH;
    uint64_t wid_frac_CXdel = pre.C_pre + (2 * FP32_MANT_WIDTH);
    uint64_t decimal_bits = max(max(wid_frac_BXdel, wid_frac_CXdel), (uint64_t)pre.A_pre);

    uint32_t test_rst = NormalizeToFP32(table_res, 254 - (0x7f + exp), decimal_bits);
    return test_rst | sign;
}