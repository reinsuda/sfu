#include "sincos_coeffs.h"
#include "utils.h"

uint32_t getTableItem(uint32_t float_hex, uint32_t delta_bits,
                      uint32_t& valid_delta) {
  // fix is 0.23
  uint32_t tid = 0;
  uint32_t exp = (float_hex >> 23) & 0xff;
  uint32_t mant = float_hex & N_BIT_1(23);
  if (exp < 123)  // exp table
  {
    tid = (mant >> delta_bits) + ((exp - 115) << 3);
    valid_delta = mant & 0xfffff;
  } else  // even table for [2^-4 ~ 2^-2)
  {
    uint32_t compensation = 127 - exp;
    uint32_t fix = (1 << 23 | mant);
    valid_delta = fix & N_BIT_1(15 + compensation);
    tid = ((fix >> (15 + compensation)) & 0xff) + 64 -
          16;  // 16 is the table index of exp==123
  }

  return tid;
}

uint32_t fp32_sin(uint32_t src, bool ftz) {
  Precision pre;
  pre.A_pre = 24;
  pre.B_pre = 18;
  pre.C_pre = 8;
  // special number handle
  uint32_t sign = src & 0x80000000;
  uint32_t nonsign = sign ^ src;
  if (fp32_is_nan(src) || fp32_is_inf(src)) {
    return 0xFFFFFFFF;
  }
  if (fp32_is_zero(src)) {
    return sign;
  }

  int32_t exp = (nonsign >> FP32_MANT_WIDTH) & N_BIT_1(FP32_EXP_WIDTH);
  uint32_t mant = nonsign & N_BIT_1(FP32_MANT_WIDTH);
  bool quad_sign_flip = false;

  // =======================================================
  // 纯位运算的象限折叠 (RTL 硬件级实现)
  // =======================================================
  // 1. 拦截精确的 1.0f (即 0x3F800000)，1.0 代表 2pi，相位于 0
  if (nonsign == 0x3F800000) {
    nonsign = 0;
    exp = 0;
    mant = 0;
  }
  // 2. 仅处理 (0.25, 1.0) 之间的浮点数折叠
  else if (nonsign > 0x3E800000) {
    // [步骤 A] Float 转 U0.32 定点数
    // 因为 nonsign > 0.25，exp 至少是 125，(exp - 118) 必定 >=
    // 7，不会出现负数移位 这一步完美提取了尾数，并放大了相应的倍数
    uint32_t V = ((1U << FP32_MANT_WIDTH) | mant) << (exp - 118);

    // [步骤 B] 纯整型定点数的象限折叠 (极其轻量)
    if (V < 0x80000000) {         // 位于 (0.25, 0.5)
      V = 0x80000000 - V;         // 相当于 0.5 - x
    } else if (V < 0xC0000000) {  // 位于 [0.5, 0.75)
      V = V - 0x80000000;         // 相当于 x - 0.5
      quad_sign_flip = true;
    } else {      // 位于 [0.75, 1.0)
      V = 0 - V;  // 相当于 1.0 - x (利用无符号溢出，等价于 2^32 - V)
      quad_sign_flip = true;
    }

    // [步骤 C] 将折叠后的定点数重新规格化 (Normalize) 回 IEEE 754
    if (V == 0) {
      // 如果 V 变成了 0（说明原输入精确等于 0.5），则折叠后的相位为 0
      nonsign = 0;
      exp = 0;
      mant = 0;
    } else {
      // 找到最高位的 1 (MSB 的位置，0~31)
      // 这里借用你之前的 leadingOne 函数
      uint32_t lo = __builtin_clz(V);

      // 计算新的指数: 如果最高位在 bit 31 (即 0.5)，对应的 exp 应该是 126
      exp = 95 + lo;  // 126 - 31 = 95

      // 清除隐藏的最高位 1
      uint32_t clear_hidden = V & ~(1U << lo);

      // 将尾数重新对齐到 23 位的 IEEE 754 标准格式
      if (lo >= FP32_MANT_WIDTH) {
        mant = clear_hidden >> (lo - FP32_MANT_WIDTH);
      } else {
        mant = clear_hidden << (FP32_MANT_WIDTH - lo);
      }

      // 重新拼接回 32bit 变量
      nonsign = (exp << FP32_MANT_WIDTH) | mant;
    }
  }

  // 计算最终的符号位
  uint32_t final_sign = sign ^ (quad_sign_flip ? 0x80000000 : 0);

  // 安全拦截查表无法处理的精确 0.25 (sin(pi/2) = 1)
  if (nonsign == 0x3E800000) {
    return 0x3F800000 | final_sign;
  }

  if (exp == 0 && ftz) {
    return sign;
  }

  if (nonsign <
      0x39868a47) {  // todo test how to set 0x39868a47 error ult is min
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

    if (exp == 115 && mant > 0x400000) { 
        rst_sig -= (5ULL << (46 - 23)); // 等效减去 5 个结果 ULP
    } else if (exp == 115) {
        rst_sig -= (2ULL << (46 - 23)); // 偏小的尾数减去 2 个 ULP
    }

    // 传递给规格化模块，注意 radix point 的位置
    // 两个 24-bit (1.23 格式) 相乘，定点在 46 位。
    return NormalizeToFP32(rst_sig, rst_exp, 2 * FP32_MANT_WIDTH);
  } else {  // TODO loop up table
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
    uint64_t table_res = SIN_fix_multi(table.c0,      // Minimax C0
                                       table.c1_abs,  // Minimax C1
                                       table.c2,      // Minimax C2
                                       delta,         // 段内相对偏移量
                                       pre.A_pre, pre.B_pre, pre.C_pre,
                                       exp,         // 当前浮点数的真实指数
                                       23,          // Mantissa 基准位宽
                                       delta_width  // 当前段的 delta 位宽
    );

    // =========================================================
    // 5. 将定点结果规格化回 FP32
    // =========================================================
    // 根据原代码的设计，定点乘加的最终输出格式假定为 0.54 (即小数点在第 54 位)
    // 并且结果对应于标准的 [0, 1] 范围，因此隐式初始指数为 127 (2^0)。
    if (table_res != 0) {
      return NormalizeToFP32(table_res, 127, 54);
    } else {
      return 0;
    }
  }
}

uint32_t fp32_cos(uint32_t src, bool ftz) {
  Precision pre;
  pre.A_pre = 24;
  pre.B_pre = 18;
  pre.C_pre = 8;
  // special number handle
  uint32_t sign = src & 0x80000000;
  uint32_t nonsign = sign ^ src;
  if (fp32_is_nan(src) || fp32_is_inf(src)) {
    return 0xFFFFFFFF;
  }
  if (fp32_is_zero(src)) {
    return sign;
  }

  int32_t exp = (nonsign >> FP32_MANT_WIDTH) & N_BIT_1(FP32_EXP_WIDTH);
  uint32_t mant = nonsign & N_BIT_1(FP32_MANT_WIDTH);
  // handle 【0.25,0.5】、【0.5,0.75】、【0.75,1】-》【0，0.25】

  // handle end

  if (exp == 0 && ftz) {
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