#include "sincos_coeffs.h"
#include "utils.h"

uint32_t getTableItem(uint32_t float_hex, uint32_t delta_bits,
                      uint32_t &valid_delta)
{
  // fix is 0.23
  uint32_t tid = 0;
  uint32_t exp = (float_hex >> 23) & 0xff;
  uint32_t mant = float_hex & N_BIT_1(23);
  if (exp < 123) // exp table
  {
    tid = (mant >> delta_bits) + ((exp - 115) << 3);
    valid_delta = mant & 0xfffff;
  }
  else // even table for [2^-4 ~ 2^-2)
  {
    uint32_t compensation = 127 - exp;
    uint32_t fix = (1 << 23 | mant);
    valid_delta = fix & N_BIT_1(15 + compensation);
    tid = ((fix >> (15 + compensation)) & 0xff) + 64 -
          16; // 16 is the table index of exp==123
  }

  return tid;
}

uint32_t fp32_sin(uint32_t src, bool ftz)
{
  Precision pre;
  pre.A_pre = 27;
  pre.B_pre = 18;
  pre.C_pre = 10;
  // special number handle
  uint32_t sign = src & 0x80000000;
  uint32_t nonsign = sign ^ src;
  if (fp32_is_nan(src) || fp32_is_inf(src))
  {
    return 0xFFFFFFFF;
  }
  if (fp32_is_zero(src))
  {
    return sign;
  }

  int32_t exp = (nonsign >> FP32_MANT_WIDTH) & N_BIT_1(FP32_EXP_WIDTH);
  uint32_t mant = nonsign & N_BIT_1(FP32_MANT_WIDTH);
  bool quad_sign_flip = false;

  // =======================================================
  // 纯位运算的象限折叠 (RTL 硬件级实现)
  // =======================================================
  // 1. 拦截精确的 1.0f (即 0x3F800000)，1.0 代表 2pi，相位于 0
  if (nonsign == 0x3F800000)
  {
    nonsign = 0;
    exp = 0;
    mant = 0;
  }
  // 2. 仅处理 (0.25, 1.0) 之间的浮点数折叠
  else if (nonsign > 0x3E800000)
  {
    uint32_t V = 0;
    int shift = (int)exp - 118;

    // [步骤 A] Float 转 U0.32 定点数 (自动实现 x mod 1.0 的神级操作)
    if (shift < 32)
    {
      // 当 shift >= 32 时 (即 exp >= 150)，浮点数已没有小数位，V 自然为 0
      // 只有 shift < 32 才移位，完美避开 C++ << 32 的 UB
      V = ((1U << FP32_MANT_WIDTH) | mant) << shift;
    }

    // [步骤 B] 纯整型定点数的象限折叠 (将任意小数完美折叠到 0 ~ 0.25)
    if (V < 0x40000000)
    {
      // [0, 0.25) -> 第一象限，本身就在目标范围内，无需操作！
    }
    else if (V < 0x80000000)
    {
      // [0.25, 0.5) -> 第二象限
      V = 0x80000000 - V; // 相当于 0.5 - x
    }
    else if (V < 0xC0000000)
    {
      // [0.5, 0.75) -> 第三象限
      V = V - 0x80000000; // 相当于 x - 0.5
      quad_sign_flip = true;
    }
    else
    {
      // [0.75, 1.0) -> 第四象限
      V = 0 - V; // 相当于 1.0 - x (利用 uint32 溢出等价于 2^32 - V)
      quad_sign_flip = true;
    }

    // [步骤 C] 将折叠后的定点数重新规格化 (Normalize) 回 IEEE 754
    if (V == 0)
    {
      // 此时说明输入刚好是 0.5/1.0/1.5... 等关键点，折叠后相位恰好为 0
      nonsign = 0;
      exp = 0;
      mant = 0;
    }
    else
    {
      // 提取前导零数量
      uint32_t lo = __builtin_clz(V);

      // 💡 修复1：正确的指数计算公式
      // 若 V=0x80000000 (0.5), lo=0, exp=126
      // 若 V=0x40000000 (0.25), lo=1, exp=125
      exp = 126 - lo;

      // 💡 修复2：极简尾数对齐法则
      // MSB 的位置在 (31 - lo)。我们想要把它下面的数据对齐到尾数的低 23 位
      // 也就是需要将数据向右平移 (31 - lo) - 23 = (8 - lo) 位
      int align_shift = 8 - (int)lo;

      if (align_shift > 0)
      {
        mant = V >> align_shift;
      }
      else
      {
        mant = V << (-align_shift);
      }

      // 使用 0x7FFFFF 掩码直接剔除隐藏位 1，告别繁琐的 bit 消除运算！
      mant &= 0x7FFFFF;

      // 重新拼接回 32bit 变量
      nonsign = (exp << FP32_MANT_WIDTH) | mant;
    }
  }

  // 计算最终的符号位
  uint32_t final_sign = sign ^ (quad_sign_flip ? 0x80000000 : 0);

  // 安全拦截查表无法处理的精确 0.25 (sin(pi/2) = 1)
  assert(nonsign < 3e800000);
  if (nonsign == 0x3E800000)
  {
    return 0x3F800000 | final_sign;
  }

  if (exp == 0 && ftz)
  {
    return sign;
  }
  if (nonsign <
      0x39868a47)
  { // todo test how to set 0x39868a47 error ult is min
    // 使用fix_multi中的乘法器和加法器做fp32的乘法
    int effective_exp = (exp == 0) ? 1 : exp;

    // 2*PI 的指数偏移是 +2
    int rst_exp = effective_exp + 2;

    // 提取 24-bit 尾数
    uint64_t sig = exp == 0 ? mant : mant | (1ULL << FP32_MANT_WIDTH);

    // 2*PI 的 24-bit 尾数 (0x40c90fdb 的尾数补上隐藏的 1)
    const uint64_t pi_sig = 0xc90fdb;

    // 执行 24bit * 24bit = 48bit 的乘法
    uint64_t rst_sig = sig * pi_sig;

    uint32_t top3 = mant >> 20; // 提取尾数的高 3 位 (0~7)

    // =========================================================
    // 💡 终极绝招：微型 ULP 补偿器 (Micro-Compensation Decoder)
    // 根据尾数高 3 位 (分为 8 个区间)，精准扣除缺失的泰勒三次项
    // RTL 实现中，这只是一个极小的 8-to-1 多路选择器，面积几乎为 0
    // =========================================================
    if (exp == 115)
    {
      // exp = 115 时的精确 ULP 补偿表
      static const uint64_t comp_115[8] = {6, 9, 12, 15, 20, 25, 31, 38};
      rst_sig -= (comp_115[top3] << 23);
    }
    else if (exp == 114)
    {
      // exp = 114 时的精确 ULP 补偿表
      static const uint64_t comp_114[8] = {2, 2, 3, 4, 5, 6, 8, 9};
      rst_sig -= (comp_114[top3] << 23);
    }

    // 传递给规格化模块，注意 radix point 的位置
    // 两个 24-bit (1.23 格式) 相乘，定点在 46 位。
    return final_sign | NormalizeToFP32(rst_sig, rst_exp, 2 * FP32_MANT_WIDTH);
  }
  else
  { // TODO loop up table
    uint32_t delta = 0;
    // 提取当前浮点数的指数部分
    uint32_t exp = (nonsign >> 23) & 0xFF;

    // 调用现有的查表寻址函数。
    // 注：第二个参数 20 是 exp < 123 时的默认 delta_bits
    uint32_t t_idx = getTableItem(nonsign, 20, delta);

    // =========================================================
    // 2. 取出对应的 Minimax 表项 (包含 C0, C1, C2)
    // =========================================================
    auto table = FP32_SINCOS_TABLE[t_idx];

    // =========================================================
    // 3. 计算当前分段的真实 Delta 位宽
    // =========================================================
    // exp < 123 采用指数分段，位宽恒定为 20。
    // exp >= 123 采用线性分段，位宽随指数变大而缩小。
    uint32_t delta_width = (exp < 123) ? 20 : (15 + (127 - exp));

    // =========================================================
    // 4. 执行定点乘加树运算 (SFU Hardware Model)
    // =========================================================
    // Minimax 的表项直接喂给原来的 SIN_fix_multi 即可。
    // 底层的硬件逻辑会自动将 raw delta 减去半区间，转换为围绕中心点的有符号
    // delta。

    uint64_t table_res = SIN_fix_multi(table.c0,     // Minimax C0
                                       table.c1_abs, // Minimax C1
                                       table.c2,     // Minimax C2
                                       delta,        // 段内相对偏移量
                                       pre.A_pre, pre.B_pre, pre.C_pre,
                                       exp,        // 当前浮点数的真实指数
                                       23,         // Mantissa 基准位宽
                                       delta_width // 当前段的 delta 位宽
    );

    uint64_t wid_frac_BXdel = (int32_t)pre.B_pre - 3 + FP32_MANT_WIDTH;
    uint64_t wid_frac_CXdel = (int32_t)pre.C_pre - 5 + FP32_MANT_WIDTH + FP32_MANT_WIDTH;

    // 算出来的 decimal_bits 必须永远等价于底层的 max_width (例如 51)
    uint64_t decimal_bits = max(max(wid_frac_BXdel, wid_frac_CXdel), (uint64_t)pre.A_pre);

    if (table_res != 0)
    {
      return final_sign | NormalizeToFP32(table_res, 127, decimal_bits);
    }
    else
    {
      return 0;
    }
  }
}

uint32_t fp32_cos(uint32_t src, bool ftz)
{
  Precision pre;
  pre.A_pre = 24;
  pre.B_pre = 18;
  pre.C_pre = 8;
  // special number handle
  uint32_t sign = src & 0x80000000;
  uint32_t nonsign = sign ^ src;
  if (fp32_is_nan(src) || fp32_is_inf(src))
  {
    return 0xFFFFFFFF;
  }
  if (fp32_is_zero(src))
  {
    return sign;
  }

  int32_t exp = (nonsign >> FP32_MANT_WIDTH) & N_BIT_1(FP32_EXP_WIDTH);
  uint32_t mant = nonsign & N_BIT_1(FP32_MANT_WIDTH);
  // handle 【0.25,0.5】、【0.5,0.75】、【0.75,1】-》【0，0.25】

  // handle end

  if (exp == 0 && ftz)
  {
    return sign;
  }

  // if (exp == 0) // denorm handle
  // {
  //     uint32_t lz = __builtin_clz(mant);

  //     // 正确的左移位数 (应为正数 1 或 2)
  //     uint32_t shift_amt = lz - 8;

  //     // 指数补偿：起始等效阶码为1，每左移一位，阶码减一
  //     exp = 9 - lz; // 等价于 exp = 1 - shift_amt;

  //     // 尾数左移，并掩去已经跑到第 23 位的那个隐含的 '1'
  //     mant = (mant << shift_amt) & N_BIT_1(FP32_MANT_WIDTH);
  // }

  // uint32_t lut_id = (mant >> (FP32_MANT_WIDTH - RCP_TABLE_BIT_WIDTH)) &
  // N_BIT_1(RCP_TABLE_BIT_WIDTH); uint32_t delta = mant &
  // N_BIT_1(FP32_MANT_WIDTH - RCP_TABLE_BIT_WIDTH); rcp_entry_t table =
  // FP32_RCP_TABLE[lut_id]; uint64_t table_res = RCP_fix_multi(table.c0,
  // table.c1_abs, table.c2, delta, pre.A_pre, pre.B_pre, pre.C_pre); uint32_t
  // test_rst = NormalizeToFP32(table_res, 254 - exp, pre.C_pre + (2 *
  // FP32_MANT_WIDTH)); return test_rst | sign;
  return 0;
}