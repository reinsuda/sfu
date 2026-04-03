#include "utils.h"
#include "rcp_coeffs.h"
uint32_t fp32_rcp(uint32_t src)
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

    uint32_t lut_id = (mant >> (FP32_MANT_WIDTH - RCP_TABLE_BIT_WIDTH)) & N_BIT_1(RCP_TABLE_BIT_WIDTH);
    uint32_t delta = mant & N_BIT_1(FP32_MANT_WIDTH - RCP_TABLE_BIT_WIDTH);
    rcp_entry_t table = FP32_RCP_TABLE[lut_id];
    uint64_t table_res = RCP_fix_multi(table.c0, table.c1_abs, table.c2, delta, pre.A_pre, pre.B_pre, pre.C_pre);
    uint32_t test_rst = NormalizeToFP32(table_res, 254 - exp, pre.C_pre + (2 * FP32_MANT_WIDTH));
    return test_rst | sign;
}