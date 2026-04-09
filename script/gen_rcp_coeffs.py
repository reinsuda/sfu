import numpy as np
import csv
import struct

def float_to_hex(f):
    """将浮点数转换为32位无符号整数表示形式(用于位运算)"""
    return struct.unpack('<I', struct.pack('<f', f))[0]

def hex_to_float(h):
    """将32位无符号整数转换为浮点数"""
    return struct.unpack('<f', struct.pack('<I', h))[0]


def compute_coeffs_rcp_128(t, p, q):
    """
    t: C0 的小数位宽 (实际硬件需预留 t+2 位以包含整数/符号)
    p: C1 的小数位宽
    q: C2 的小数位宽
    针对 RCP (1/x) 在 [1, 2) 区间的分段优化 (128段)
    """
    errmax = 0
    num_segments = 128
    dx_max = 1 / 128
    results = []

    print(f"{'Segment':<8} | {'C0 (Hex)':<12} | {'|C1| (Hex)':<12} | {'C2 (Hex)':<12} | {'Error':<10}")
    print("-" * 75)

    for i in range(num_segments):
        # 1. 确定当前段的【中心点】(对应硬件的 reference point is in the middle)
        m_center = 1.0 + i * dx_max + (dx_max / 2.0)
        
        # 2. 采样范围：必须是围绕中心点的相对偏移量 delta [-dx_max/2, dx_max/2]
        n = 2**16 
        delta_nodes = np.linspace(-dx_max / 2.0, dx_max / 2.0, n)
        
        # 3. 计算真实的 Y 值：中心点 + delta
        y_nodes = 1.0 / (m_center + delta_nodes)

        # 2. 初始拟合
        poly_coeffs = np.polyfit(delta_nodes, y_nodes, 2)
        a2_raw, a1_raw = poly_coeffs[0], poly_coeffs[1]

        # 3. 独立量化 C1 和 C2（删除所有冗余的交叉补偿逻辑）
        C1 = np.round(a1_raw * (2**p)) * (2**-p)
        C2 = np.round(a2_raw * (2**q)) * (2**-q) # 直接四舍五入，注意是负 q！

        # 4. 平衡常数项 C0
        rem_y = 1.0 / (m_center + delta_nodes) - (C1 * delta_nodes + C2 * (delta_nodes**2))
        a0_minimax = (np.max(rem_y) + np.min(rem_y)) / 2
        C0 = np.round(a0_minimax * (2**t)) * (2**-t)

        # 7. 误差计算 (测试集也必须是对称的 delta 范围)
        test_delta = np.linspace(-dx_max / 2.0, dx_max / 2.0, 500)
        actual_y = 1.0 / (m_center + test_delta)
        approx_y = C0 + C1 * test_delta + C2 * (test_delta**2)
        err = np.max(np.abs(actual_y - approx_y))

        if err > errmax:
            errmax = err
        
        c0_int = int(round(C0 * 2**t))
        c1_int = int(round(C1 * 2**p))
        c2_int = int(round(C2 * 2**q))

        results.append({
            "segment": i,
            "C0": C0, "C1": C1, "C2": C2,
            "C0_int": c0_int,
            "C1_int": c1_int,
            "C2_int": c2_int,
            "err": err
        })

        if i < 3 or i == 127:
            # 修复截断：掩码扩展 2 个 bits，以容纳 1.0 的整数位
            mask_t = (1 << (t + 2)) - 1
            mask_p = (1 << (p + 2)) - 1
            mask_q = (1 << (q + 2)) - 1

            c0_display = c0_int & mask_t
            
            # C1 取绝对值，转为正数显示
            c1_display = abs(c1_int) & mask_p 
            
            c2_display = c2_int & mask_q
            
            # 格式化打印：使用动态补零以对齐
            print(f"{i:<8} | {c0_display:<12x} | {c1_display:<12x} | {c2_display:<12x} | {err:.2e}")

    good_bits = np.abs(np.log2(errmax))
    print("-" * 75)
    print(f"最大绝对误差 (MAE): {errmax:.12e}")
    print(f"等效精度 (Good Bits): {good_bits:.2f} bits\n")
    
    return results

def save_to_files(table, t, p, q, name, table_name, table_size):
    # 1. 保存为 C++ Header (C1 修改为正数保存)
    with open(name, "w") as f:
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include \"utils.h\"\n\n")
        f.write(f"// Configuration: C0={t}bits, C1={p}bits(Abs Value), C2={q}bits\n")
        f.write("// NOTE: C1 is stored as a POSITIVE value. Use SUBTRACTION in hardware.\n\n")
        
        
        f.write(f"const rcp_entry_t {table_name}[{table_size}] = {{\n")
        
        for r in table:
            # C0 保持不变
            c0_hex = r['C0_int'] & ((1 << t) - 1)
            
            # C1 取绝对值 (转换为正数)
            c1_abs = abs(r['C1_int'])
            c1_hex = c1_abs & ((1 << p) - 1)
            
            # C2 保持补码形式 (因为 C2 可能跨越正负)
            c2_hex = r['C2_int'] & ((1 << q) - 1)
            
            f.write(f"    {{ 0x{c0_hex:08x}, 0x{c1_hex:04x}, 0x{c2_hex:03x} }}, \n")
            
        f.write("};\n\n")

    # 2. 同步更新 Verilog Hex 文件
    with open("rcp_coeffs.hex", "w") as f:
        for r in table:
            c0_h = r['C0_int'] & ((1 << t) - 1)
            c1_h = abs(r['C1_int']) & ((1 << p) - 1) # 存绝对值
            c2_h = r['C2_int'] & ((1 << q) - 1)
            combined = (c0_h << (p + q)) | (c1_h << q) | c2_h
            f.write(f"{combined:x}\n")

    print("文件已更新：C1 现在以正数（绝对值）16 进制格式保存。")
# --- 主程序执行 ---
# 根据你的需求：t=27 (精度高), p=12, q=8
t_width, p_width, q_width = 24, 18, 8
final_table = compute_coeffs_rcp_128(t_width, p_width, q_width)
save_to_files(final_table, t_width+1, p_width+1, q_width+1,"rcp_coeffs.h", "FP32_RCP_TABLE",128)


def compute_coeffs_sqrt_64e(t, p, q):
    errmax = 0
    # 修改1: 统一将段数修正为 128，与函数名以及末尾的 i==127 相匹配
    num_segments = 64
    dx_max = 1 / 64
    results = []

    # 修改2: sqrt 的 C1 是正数，C2 是负数，因此移除了原本表头里 C1 的绝对值符号 |C1|
    print(f"{'Segment':<8} | {'C0 (Hex)':<12} | {'C1 (Hex)':<12} | {'C2 (Hex)':<12} | {'Error':<10}")
    print("-" * 75)

    for i in range(num_segments):
        # 1. 确定当前段的【中心点】(对应硬件的 reference point is in the middle)
        m_center = 1.0 + i * dx_max + (dx_max / 2.0)
        
        # 2. 采样范围：必须是围绕中心点的相对偏移量 delta [-dx_max/2, dx_max/2]
        n = 2**16 
        delta_nodes = np.linspace(-dx_max / 2.0, dx_max / 2.0, n)
        
        # 3. 修改3: 计算真实的 Y 值 (改为 np.sqrt)
        y_nodes = np.sqrt(m_center + delta_nodes)

        # 4. 初始拟合
        poly_coeffs = np.polyfit(delta_nodes, y_nodes, 2)
        a2_raw, a1_raw = poly_coeffs[0], poly_coeffs[1]

        # 5. 独立量化 C1 和 C2
        C1 = np.round(a1_raw * (2**p)) * (2**-p)
        C2 = np.round(a2_raw * (2**q)) * (2**-q)

        # 6. 修改4: 平衡常数项 C0 (使用 np.sqrt)
        rem_y = np.sqrt(m_center + delta_nodes) - (C1 * delta_nodes + C2 * (delta_nodes**2))
        a0_minimax = (np.max(rem_y) + np.min(rem_y)) / 2
        C0 = np.round(a0_minimax * (2**t)) * (2**-t)

        # 7. 修改5: 误差计算 (使用 np.sqrt)
        test_delta = np.linspace(-dx_max / 2.0, dx_max / 2.0, 500)
        actual_y = np.sqrt(m_center + test_delta)
        approx_y = C0 + C1 * test_delta + C2 * (test_delta**2)
        err = np.max(np.abs(actual_y - approx_y))

        if err > errmax:
            errmax = err
        
        c0_int = int(round(C0 * 2**t))
        c1_int = int(round(C1 * 2**p))
        c2_int = int(round(C2 * 2**q))

        results.append({
            "segment": i,
            "C0": C0, "C1": C1, "C2": C2,
            "C0_int": c0_int,
            "C1_int": c1_int,
            "C2_int": abs(c2_int),
            "err": err
        })

        if i < 3 or i == 127:
            # 修复截断：掩码扩展 2 个 bits，以容纳 1.0 的整数位及符号位
            mask_t = (1 << (t + 2)) - 1
            mask_p = (1 << (p + 2)) - 1
            mask_q = (1 << (q + 2)) - 1

            c0_display = c0_int & mask_t
            
            # 修改6: 倒数(1/x)的导数是负的，所以原代码用了 abs()。
            # 而对于 sqrt(x)，一阶导数(C1)是正的，二阶导数(C2)是负的。
            # 这里统一改回标准二进制补码打印，去掉 abs()。
            c1_display = c1_int & mask_p 
            c2_display = c2_int & mask_q
            
            # 格式化打印：使用动态补零以对齐
            print(f"{i:<8} | {c0_display:<12x} | {c1_display:<12x} | {c2_display:<12x} | {err:.2e}")

    good_bits = np.abs(np.log2(errmax))
    print("-" * 75)
    print(f"最大绝对误差 (MAE): {errmax:.12e}")
    print(f"等效精度 (Good Bits): {good_bits:.2f} bits\n")
    
    return results


t_width, p_width, q_width = 27, 18, 10
final_table = compute_coeffs_sqrt_64e(t_width, p_width, q_width)
save_to_files(final_table, t_width+1, p_width+1, q_width+1,"sqrte_coeffs.h", "FP32_SQRT_E_TABLE",64)

def compute_coeffs_sqrt_64o(t, p, q):
    errmax = 0
    num_segments = 64
    dx_max = 1 / 64
    results = []

    print(f"{'Segment':<8} | {'C0 (Hex)':<12} | {'C1 (Hex)':<12} | {'C2 (Hex)':<12} | {'Error':<10}")
    print("-" * 75)

    for i in range(num_segments):
        # 1. 确定当前段的【中心点】
        m_center = 1.0 + i * dx_max + (dx_max / 2.0)
        
        # 2. 采样范围
        delta_nodes = np.linspace(-dx_max / 2.0, dx_max / 2.0, 2**16)
        
        # 3. 终极内核：泰勒展开直接求导 (针对 sqrt(2x))
        # f(x) = sqrt(2x)
        # 一阶导 f'(x) = 1 / sqrt(2x)
        a1_raw = 1.0 / np.sqrt(2.0 * m_center)
        
        # 二阶导 f''(x) = -1 / (x * sqrt(8x))
        # 泰勒二次项 = f''(x)/2 = -1 / (4 * x * sqrt(2x))
        a2_raw = -1.0 / (4.0 * m_center * np.sqrt(2.0 * m_center))

        # 4. 独立量化 C1 和 C2
        C1 = np.round(a1_raw * (2**p)) * (2**-p)
        C2 = np.round(a2_raw * (2**q)) * (2**-q)

        # 5. 【已修复 Bug 1】平衡常数项 C0 (必须计算 sqrt(2x) 的残差)
        rem_y = np.sqrt((m_center + delta_nodes) * 2.0) - (C1 * delta_nodes + C2 * (delta_nodes**2))
        a0_minimax = (np.max(rem_y) + np.min(rem_y)) / 2
        C0 = np.round(a0_minimax * (2**t)) * (2**-t)

        # 6. 【已修复 Bug 2】误差计算 (也必须验证 sqrt(2x))
        test_delta = np.linspace(-dx_max / 2.0, dx_max / 2.0, 500)
        actual_y = np.sqrt((m_center + test_delta) * 2.0)
        approx_y = C0 + C1 * test_delta + C2 * (test_delta**2)
        err = np.max(np.abs(actual_y - approx_y))

        if err > errmax:
            errmax = err
        
        c0_int = int(round(C0 * 2**t))
        c1_int = int(round(C1 * 2**p))
        c2_int = int(round(C2 * 2**q))

        results.append({
            "segment": i,
            "C0": C0, "C1": C1, "C2": C2,
            "C0_int": c0_int,
            "C1_int": c1_int,
            "C2_int": abs(c2_int), # 硬件做减法，统一导出绝对值
            "err": err
        })

        # 7. 【已修复 Bug 3】64 段的末尾是 63，不是 127
        if i < 3 or i == 63:
            mask_t = (1 << (t + 2)) - 1
            mask_p = (1 << (p + 2)) - 1
            mask_q = (1 << (q + 2)) - 1

            c0_display = c0_int & mask_t
            c1_display = c1_int & mask_p 
            
            # 打印展示时也直接看绝对值，防止被补码迷惑
            c2_display = abs(c2_int) & mask_q 
            
            print(f"{i:<8} | {c0_display:<12x} | {c1_display:<12x} | {c2_display:<12x} | {err:.2e}")

    good_bits = np.abs(np.log2(errmax))
    print("-" * 75)
    print(f"最大绝对误差 (MAE): {errmax:.12e}")
    print(f"等效精度 (Good Bits): {good_bits:.2f} bits\n")
    
    return results

t_width, p_width, q_width = 27, 18, 10
final_table = compute_coeffs_sqrt_64o(t_width, p_width, q_width)
save_to_files(final_table, t_width+1, p_width+1, q_width+1,"sqrto_coeffs.h", "FP32_SQRT_O_TABLE",64)

# ==========================================
# 偶数表 (Even Table): 针对 1 / sqrt(x)
# ==========================================
def compute_coeffs_rsqrt_64e(t, p, q):
    errmax = 0
    num_segments = 64
    dx_max = 1 / 64
    results = []

    print(f"RSQRT Even Table (1/sqrt(x))")
    print(f"{'Segment':<8} | {'C0 (Hex)':<12} | {'C1 (Hex)':<12} | {'C2 (Hex)':<12} | {'Error':<10}")
    print("-" * 75)

    for i in range(num_segments):
        m_center = 1.0 + i * dx_max + (dx_max / 2.0)
        delta_nodes = np.linspace(-dx_max / 2.0, dx_max / 2.0, 2**16)
        
        # 泰勒展开内核: f(x) = x^(-0.5)
        # 一阶导 f'(x) = -0.5 * x^(-1.5)
        a1_raw = -0.5 / (m_center**1.5)
        
        # 二阶导 f''(x) = 0.75 * x^(-2.5)
        # 泰勒二次项 = f''(x)/2 = 0.375 * x^(-2.5)
        a2_raw = 0.375 / (m_center**2.5)

        C1 = np.round(a1_raw * (2**p)) * (2**-p)
        C2 = np.round(a2_raw * (2**q)) * (2**-q)

        # 残差计算: 1 / sqrt(x)
        rem_y = (1.0 / np.sqrt(m_center + delta_nodes)) - (C1 * delta_nodes + C2 * (delta_nodes**2))
        a0_minimax = (np.max(rem_y) + np.min(rem_y)) / 2
        C0 = np.round(a0_minimax * (2**t)) * (2**-t)

        test_delta = np.linspace(-dx_max / 2.0, dx_max / 2.0, 500)
        actual_y = 1.0 / np.sqrt(m_center + test_delta)
        approx_y = C0 + C1 * test_delta + C2 * (test_delta**2)
        err = np.max(np.abs(actual_y - approx_y))

        if err > errmax:
            errmax = err
        
        c0_int = int(round(C0 * 2**t))
        c1_int = int(round(C1 * 2**p))
        c2_int = int(round(C2 * 2**q))

        results.append({
            "segment": i,
            "C0": C0, "C1": C1, "C2": C2,
            "C0_int": c0_int,
            "C1_int": abs(c1_int), # 硬件做减法，导出绝对值
            "C2_int": abs(c2_int), 
            "err": err
        })

        if i < 3 or i == 63:
            mask_t = (1 << (t + 2)) - 1
            mask_p = (1 << (p + 2)) - 1
            mask_q = (1 << (q + 2)) - 1

            c0_display = c0_int & mask_t
            c1_display = abs(c1_int) & mask_p 
            c2_display = abs(c2_int) & mask_q 
            
            print(f"{i:<8} | {c0_display:<12x} | {c1_display:<12x} | {c2_display:<12x} | {err:.2e}")

    good_bits = np.abs(np.log2(errmax))
    print("-" * 75)
    print(f"最大绝对误差 (MAE): {errmax:.12e}")
    print(f"等效精度 (Good Bits): {good_bits:.2f} bits\n")
    
    return results


# ==========================================
# 奇数表 (Odd Table): 针对 1 / sqrt(2x)
# ==========================================
def compute_coeffs_rsqrt_64o(t, p, q):
    errmax = 0
    num_segments = 64
    dx_max = 1 / 64
    results = []

    print(f"RSQRT Odd Table (1/sqrt(2x))")
    print(f"{'Segment':<8} | {'C0 (Hex)':<12} | {'C1 (Hex)':<12} | {'C2 (Hex)':<12} | {'Error':<10}")
    print("-" * 75)

    for i in range(num_segments):
        m_center = 1.0 + i * dx_max + (dx_max / 2.0)
        delta_nodes = np.linspace(-dx_max / 2.0, dx_max / 2.0, 2**16)
        
        # 泰勒展开内核: f(x) = (2x)^(-0.5)
        # 一阶导 f'(x) = -0.5 / (sqrt(2) * x^1.5)
        a1_raw = -0.5 / (np.sqrt(2.0) * (m_center**1.5))
        
        # 二阶导 f''(x) = 0.75 / (sqrt(2) * x^2.5)
        # 泰勒二次项 = f''(x)/2 = 0.375 / (sqrt(2) * x^2.5)
        a2_raw = 0.375 / (np.sqrt(2.0) * (m_center**2.5))

        C1 = np.round(a1_raw * (2**p)) * (2**-p)
        C2 = np.round(a2_raw * (2**q)) * (2**-q)

        # 残差计算: 1 / sqrt(2x)
        rem_y = (1.0 / np.sqrt((m_center + delta_nodes) * 2.0)) - (C1 * delta_nodes + C2 * (delta_nodes**2))
        a0_minimax = (np.max(rem_y) + np.min(rem_y)) / 2
        C0 = np.round(a0_minimax * (2**t)) * (2**-t)

        test_delta = np.linspace(-dx_max / 2.0, dx_max / 2.0, 500)
        actual_y = 1.0 / np.sqrt((m_center + test_delta) * 2.0)
        approx_y = C0 + C1 * test_delta + C2 * (test_delta**2)
        err = np.max(np.abs(actual_y - approx_y))

        if err > errmax:
            errmax = err
        
        c0_int = int(round(C0 * 2**t))
        c1_int = int(round(C1 * 2**p))
        c2_int = int(round(C2 * 2**q))

        results.append({
            "segment": i,
            "C0": C0, "C1": C1, "C2": C2,
            "C0_int": c0_int,
            "C1_int": abs(c1_int), # 硬件做减法，导出绝对值
            "C2_int": abs(c2_int),
            "err": err
        })

        if i < 3 or i == 63:
            mask_t = (1 << (t + 2)) - 1
            mask_p = (1 << (p + 2)) - 1
            mask_q = (1 << (q + 2)) - 1

            c0_display = c0_int & mask_t
            c1_display = abs(c1_int) & mask_p 
            c2_display = abs(c2_int) & mask_q 
            
            print(f"{i:<8} | {c0_display:<12x} | {c1_display:<12x} | {c2_display:<12x} | {err:.2e}")

    good_bits = np.abs(np.log2(errmax))
    print("-" * 75)
    print(f"最大绝对误差 (MAE): {errmax:.12e}")
    print(f"等效精度 (Good Bits): {good_bits:.2f} bits\n")
    
    return results

# ==========================================
# 导出表格调用示例
# ==========================================
t_width, p_width, q_width = 27, 18, 10

final_table_e = compute_coeffs_rsqrt_64e(t_width, p_width, q_width)
save_to_files(final_table_e, t_width+1, p_width+1, q_width+1, "rsqrte_coeffs.h", "FP32_RSQRT_E_TABLE", 64)

final_table_o = compute_coeffs_rsqrt_64o(t_width, p_width, q_width)
save_to_files(final_table_o, t_width+1, p_width+1, q_width+1, "rsqrto_coeffs.h", "FP32_RSQRT_O_TABLE", 64)



def compute_coeffs_log2_128(t, p, q):
    """
    t: C0 的小数位宽
    p: C1 的小数位宽
    q: C2 的小数位宽
    针对 LOG2(x) 在 [1, 2) 区间的分段优化 (128段)
    """
    errmax = 0
    num_segments = 128
    dx_max = 1 / 128
    results = []

    # LOG2 的一阶导数是正的，二阶导数是负的 (同 SQRT)，这里去掉 C1 的绝对值符号
    print(f"{'Segment':<8} | {'C0 (Hex)':<12} | {'C1 (Hex)':<12} | {'C2 (Hex)':<12} | {'Error':<10}")
    print("-" * 75)

    for i in range(num_segments):
        # 1. 确定当前段的【中心点】
        m_center = 1.0 + i * dx_max + (dx_max / 2.0)
        
        # 2. 采样范围
        delta_nodes = np.linspace(-dx_max / 2.0, dx_max / 2.0, 2**16)
        
        # 3. 终极内核：泰勒展开直接求导 (针对 log2(x))
        # f(x) = log2(x) = ln(x) / ln(2)
        # 一阶导 f'(x) = 1 / (x * ln(2))
        a1_raw = 1.0 / (m_center * np.log(2.0))
        
        # 二阶导 f''(x) = -1 / (x^2 * ln(2))
        # 泰勒二次项 = f''(x)/2 = -1 / (2 * x^2 * ln(2))
        a2_raw = -1.0 / (2.0 * (m_center**2) * np.log(2.0))

        # 4. 独立量化 C1 和 C2
        C1 = np.round(a1_raw * (2**p)) * (2**-p)
        C2 = np.round(a2_raw * (2**q)) * (2**-q)

        # 5. 平衡常数项 C0
        if i == 0:
            # 【终极必杀技：零点锚定】
            # 对于第 0 段，强制让多项式在 x=1.0 (即 delta = -dx_max/2) 处的结果绝对等于 0
            left_delta = -dx_max / 2.0
            # 既然实际要求 C0 + C1*delta + C2*delta^2 = 0，那么：
            C0_exact = -(C1 * left_delta + C2 * (left_delta**2))
            C0 = np.round(C0_exact * (2**t)) * (2**-t)
        else:
            # 其他段依然使用 minimax 保证全局绝对误差平均分布
            rem_y = np.log2(m_center + delta_nodes) - (C1 * delta_nodes + C2 * (delta_nodes**2))
            a0_minimax = (np.max(rem_y) + np.min(rem_y)) / 2
            C0 = np.round(a0_minimax * (2**t)) * (2**-t)

        # 6. 误差计算
        test_delta = np.linspace(-dx_max / 2.0, dx_max / 2.0, 500)
        actual_y = np.log2(m_center + test_delta)
        approx_y = C0 + C1 * test_delta + C2 * (test_delta**2)
        err = np.max(np.abs(actual_y - approx_y))

        if err > errmax:
            errmax = err
        
        c0_int = int(round(C0 * 2**t))
        c1_int = int(round(C1 * 2**p))
        c2_int = int(round(C2 * 2**q))

        results.append({
            "segment": i,
            "C0": C0, "C1": C1, "C2": C2,
            "C0_int": c0_int,
            "C1_int": c1_int,          # LOG2 的一阶导是正数，直接原样保存
            "C2_int": abs(c2_int),     # 强制转换为绝对值！避免底层减补码再次发生爆炸
            "err": err
        })

        if i < 3 or i == 127:
            # 修复截断：掩码扩展 2 个 bits，以容纳 1.0 的整数位及符号位
            mask_t = (1 << (t + 2)) - 1
            mask_p = (1 << (p + 2)) - 1
            mask_q = (1 << (q + 2)) - 1

            c0_display = c0_int & mask_t
            c1_display = c1_int & mask_p 
            
            # C2 直接打印绝对值
            c2_display = abs(c2_int) & mask_q 
            
            print(f"{i:<8} | {c0_display:<12x} | {c1_display:<12x} | {c2_display:<12x} | {err:.2e}")

    good_bits = np.abs(np.log2(errmax))
    print("-" * 75)
    print(f"最大绝对误差 (MAE): {errmax:.12e}")
    print(f"等效精度 (Good Bits): {good_bits:.2f} bits\n")
    
    return results

# ==========================================
# 导出表格调用示例
# ==========================================
t_width, p_width, q_width = 27, 18, 10
final_table = compute_coeffs_log2_128(t_width, p_width, q_width)
save_to_files(final_table, t_width+1, p_width+1, q_width+1, "log2_coeffs.h", "FP32_LOG2_TABLE", 128)


def compute_sin_coeffs_minimax(t_bits, p_bits, q_bits):
    """
    基于 Minimax 原理生成 SIN 查找表系数。
    t_bits: C0 (fixa) 的小数位宽 (对应 C++ 中的 A_pre)
    p_bits: C1 (fixb) 的小数位宽 (对应 C++ 中的 B_pre - 3 或等效偏移)
    q_bits: C2 (fixc) 的小数位宽 (对应 C++ 中的 C_pre - 5 或等效偏移)
    """
    segments = []
    
    # ==============================================================
    # 1. 重构 C++ 中的分段逻辑，获取每段的中心点(x_c)和区间宽度(dx)
    # ==============================================================
    
    # 区域 1: 浮点数指数区段 [2^-12, 2^-4) -> exp 115 to 122
    for exp in range(115, 123):
        for partId in range(8):
            offset = partId << 20
            # C++ 代码通过将 bit19 置 1 恰好获取了该子段的中心点
            float_hex = (exp << 23) | offset | (1 << 19)
            x_c = hex_to_float(float_hex)
            
            # 该指数下整个区间的宽度为 2^(exp-127)，分为8个子段，每段宽度为 2^(exp-130)
            dx = 2**(exp - 130) 
            segments.append((x_c, dx, f"exp{exp}_p{partId}"))

    # 区域 2: 线性均匀分段区段 [2^-4, 2^-2) -> [0.0625, 0.25)
    border = 127 - 123 # = 4
    x_left = 1.0 / (1 << border)  # 2^-4 = 1/16 = 0.0625
    dx = 1.0 / 256.0              # 线性步长
    while x_left < 0.25 - 1e-6:   # 0.25 is 2^-2 (加一个小偏移防止精度丢失导致多跑一轮)
        x_c = x_left + dx / 2.0   # C++ 中的 x_left + 1/512 恰好是中心点
        segments.append((x_c, dx, f"lin_{x_left:.4f}"))
        x_left += dx

    # ==============================================================
    # 2. Minimax 拟合与量化
    # ==============================================================
    errmax = 0
    results = []

    print(f"{'Segment':<12} | {'x_center':<10} | {'C0 (Hex)':<10} | {'|C1| (Hex)':<10} | {'|C2| (Hex)':<10} | {'Error':<10}")
    print("-" * 80)

    for i, (x_c, dx, label) in enumerate(segments):
        # 1. 采样范围
        n = 2**15 
        delta_nodes = np.linspace(-dx / 2.0, dx / 2.0, n)
        
        # 2. 计算真实的 Y 值：sin(2 * pi * (x_c + delta))
        y_nodes = np.sin(np.pi * 2 * (x_c + delta_nodes))

        # 3. 初始多项式拟合
        poly_coeffs = np.polyfit(delta_nodes, y_nodes, 2)
        a2_raw, a1_raw = poly_coeffs[0], poly_coeffs[1]

        # ==============================================================
        # 💡 核心修复：计算纯小数位宽 (Fractional Bits)
        # ==============================================================
        t_frac = t_bits          # C0 是纯小数，小数位 = 总位宽 = 27
        p_frac = p_bits - 3      # C1 约等于 6.28，需要 3 位整数，小数位 = 18 - 3 = 15
        q_frac = q_bits - 5      # C2 约等于 19.7，需要 5 位整数，小数位 = 10 - 5 = 5 (若 q_bits=8，则为 3)

        # 4. 独立量化 C1 和 C2 (必须使用小数位宽 t_frac / p_frac / q_frac！)
        C1 = np.round(a1_raw * (2**p_frac)) * (2**-p_frac)
        C2 = np.round(a2_raw * (2**q_frac)) * (2**-q_frac)

        # 5. 平衡常数项 C0
        rem_y = y_nodes - (C1 * delta_nodes + C2 * (delta_nodes**2))
        a0_minimax = (np.max(rem_y) + np.min(rem_y)) / 2.0
        C0 = np.round(a0_minimax * (2**t_frac)) * (2**-t_frac)

        # 6. 误差计算
        test_delta = np.linspace(-dx / 2.0, dx / 2.0, 1000)
        actual_y = np.sin(np.pi * 2 * (x_c + test_delta))
        approx_y = C0 + C1 * test_delta + C2 * (test_delta**2)
        err = np.max(np.abs(actual_y - approx_y))

        if err > errmax:
            errmax = err

        # 7. 转为整数表示以生成 Hex (同样必须乘以 2 的小数位宽次方！)
        c0_int = int(round(abs(C0) * (2**t_frac)))
        c1_int = int(round(abs(C1) * (2**p_frac)))
        c2_int = int(round(abs(C2) * (2**q_frac)))

        results.append({
            "label": label, "x_c": x_c,
            "C0_int": c0_int, "C1_int": c1_int, "C2_int": c2_int,
            "err": err
        })

        # 打印头3个和最后3个作为观察对照
        if i < 3 or i > len(segments) - 4:
            # 打印用的掩码使用传入的总位宽
            mask_t = (1 << t_bits) - 1
            mask_p = (1 << p_bits) - 1
            mask_q = (1 << q_bits) - 1

            c0_disp = c0_int & mask_t
            c1_disp = c1_int & mask_p
            c2_disp = c2_int & mask_q

            print(f"{label:<12} | {x_c:<10.6f} | 0x{c0_disp:<08x} | 0x{c1_disp:<08x} | 0x{c2_disp:<08x} | {err:.2e}")
        
        if i == 3:
            print(f"{'...':<12} | {'...':<10} | {'...':<10} | {'...':<10} | {'...':<10} | {'...':<10}")

    good_bits = np.abs(np.log2(errmax))
    print("-" * 80)
    print(f"Total Segments: {len(segments)}")
    print(f"最大绝对误差 (MAE): {errmax:.12e}")
    print(f"等效精度 (Good Bits): {good_bits:.2f} bits\n")

    return results

t_width, p_width, q_width = 27, 18, 10
final_table = compute_sin_coeffs_minimax(t_width, p_width, q_width)
save_to_files(final_table, t_width+1, p_width+1, q_width+1, "sincos_coeffs.h", "FP32_SINCOS_TABLE", 128)


def compute_coeffs_exp_32(t=25, p=19, q=13):
    """
    t: C0 的小数位宽 (对应你 C++ 里的 pre.A_pre = 25)
    p: C1 的小数位宽 (对应 pre.B_pre = 19)
    q: C2 的小数位宽 (对应 pre.C_pre = 13)
    
    生成 2^x 在 [0, 1) 区间的 32 分段查找表
    这正是处理 e^x 从 -8 到 -INF 长尾所必须的底层硬件表！
    """
    errmax = 0
    num_segments = 32
    dx_max = 1.0 / 32.0
    results = []

    print(f"{'Segment':<8} | {'C0 (Hex)':<10} | {'C1 (Hex)':<10} | {'C2 (Hex)':<10} | {'Error':<10}")
    print("-" * 65)

    for i in range(num_segments):
        # 1. 确定当前段的【中心点】(0 ~ 1 之间)
        m_center = i * dx_max + (dx_max / 2.0)
        
        # 2. 采样范围：对称的 delta [-dx_max/2, dx_max/2]
        n = 2**16 
        delta_nodes = np.linspace(-dx_max / 2.0, dx_max / 2.0, n)
        
        # 3. 计算真实的 Y 值：2^(center + delta)
        y_nodes = np.exp2(m_center + delta_nodes)

        # 4. 二次多项式初始拟合
        poly_coeffs = np.polyfit(delta_nodes, y_nodes, 2)
        a2_raw, a1_raw = poly_coeffs[0], poly_coeffs[1]

        # 5. 独立量化 C1 和 C2
        C1 = np.round(a1_raw * (2**p)) * (2**-p)
        C2 = np.round(a2_raw * (2**q)) * (2**-q)

        # 6. 平衡常数项 C0 (Minimax 中心化补偿)
        rem_y = np.exp2(m_center + delta_nodes) - (C1 * delta_nodes + C2 * (delta_nodes**2))
        a0_minimax = (np.max(rem_y) + np.min(rem_y)) / 2
        C0 = np.round(a0_minimax * (2**t)) * (2**-t)

        # 7. 误差计算
        test_delta = np.linspace(-dx_max / 2.0, dx_max / 2.0, 500)
        actual_y = np.exp2(m_center + test_delta)
        approx_y = C0 + C1 * test_delta + C2 * (test_delta**2)
        err = np.max(np.abs(actual_y - approx_y))

        if err > errmax:
            errmax = err
        
        # 转化为定点数整数值
        c0_int = int(round(C0 * 2**t))
        c1_int = int(round(C1 * 2**p))
        c2_int = int(round(C2 * 2**q))

        results.append({
            "segment": i,
            "C0": C0, "C1": C1, "C2": C2,
            "C0_int": c0_int,
            "C1_int": c1_int,
            "C2_int": c2_int,
            "err": err
        })

        # ====================================================
        # 掩码提取：
        # C0 的范围是 1.0 ~ 2.0，必须保留至少 2 个整数位
        # C1 (斜率) 约等于 ln(2) * 2^x，最大约 1.38，也需整数位
        # ====================================================
        mask_t = (1 << (t + 2)) - 1
        mask_p = (1 << (p + 2)) - 1
        mask_q = (1 << (q + 2)) - 1

        c0_display = c0_int & mask_t
        c1_display = abs(c1_int) & mask_p 
        c2_display = c2_int & mask_q
        
        # 打印头尾几项看看结果
        if i < 3 or i >= 29:
            print(f"{i:<8} | {c0_display:<10x} | {c1_display:<10x} | {c2_display:<10x} | {err:.2e}")
        elif i == 3:
            print(f"{'...':<8} | {'...':<10} | {'...':<10} | {'...':<10} | {'...':<10}")

    good_bits = np.abs(np.log2(errmax))
    print("-" * 65)
    print(f"最大绝对误差 (MAE): {errmax:.12e}")
    print(f"等效精度 (Good Bits): {good_bits:.2f} bits\n")
    
    return results

t_width, p_width, q_width = 24, 18, 13
final_table = compute_coeffs_exp_32(t_width, p_width, q_width)
save_to_files(final_table, t_width+1, p_width+1, q_width+1, "exp_coeffs.h", "FP32_EXP_TABLE", 32)


def compute_coeffs_exp2_128(t, p, q):
    """
    t: C0 的小数位宽 (对应你 C++ 代码中的 A_pre 实际小数位)
    p: C1 的小数位宽 (对应 B_pre)
    q: C2 的小数位宽 (对应 C_pre)
    针对 2^x 在 [0, 1) 区间的分段优化 (128段)
    """
    errmax = 0
    num_segments = 128
    dx_max = 1.0 / 128.0
    results = []

    print(f"{'Segment':<8} | {'C0 (Hex)':<12} | {'C1 (Hex)':<12} | {'C2 (Hex)':<12} | {'Error':<10}")
    print("-" * 75)

    for i in range(num_segments):
        # 1. 确定当前段的【中心点】
        # 2^x 的区间是 [0, 1)，起点为 0.0
        m_center = 0.0 + i * dx_max + (dx_max / 2.0)
        
        # 2. 采样范围：围绕中心点的相对偏移量 delta [-dx_max/2, dx_max/2]
        n = 2**16 
        delta_nodes = np.linspace(-dx_max / 2.0, dx_max / 2.0, n)
        
        # 3. 计算真实的 Y 值：2^(中心点 + delta)
        y_nodes = 2.0 ** (m_center + delta_nodes)

        # 4. 初始拟合 (二阶多项式)
        # 相当于找到最佳的 a2*x^2 + a1*x + a0
        poly_coeffs = np.polyfit(delta_nodes, y_nodes, 2)
        a2_raw, a1_raw = poly_coeffs[0], poly_coeffs[1]

        # 5. 独立量化 C1 和 C2
        # 注意：2^x 的一阶和二阶导数均为正，直接四舍五入即可
        C1 = np.round(a1_raw * (2**p)) * (2**-p)
        C2 = np.round(a2_raw * (2**q)) * (2**-q) 

        # 6. 平衡常数项 C0 (Minimax 误差平衡)
        rem_y = (2.0 ** (m_center + delta_nodes)) - (C1 * delta_nodes + C2 * (delta_nodes**2))
        a0_minimax = (np.max(rem_y) + np.min(rem_y)) / 2.0
        C0 = np.round(a0_minimax * (2**t)) * (2**-t)

        # 7. 误差计算 (在测试集上验证)
        test_delta = np.linspace(-dx_max / 2.0, dx_max / 2.0, 500)
        actual_y = 2.0 ** (m_center + test_delta)
        approx_y = C0 + C1 * test_delta + C2 * (test_delta**2)
        err = np.max(np.abs(actual_y - approx_y))

        if err > errmax:
            errmax = err
        
        c0_int = int(round(C0 * 2**t))
        c1_int = int(round(C1 * 2**p))
        c2_int = int(round(C2 * 2**q))

        results.append({
            "segment": i,
            "C0": C0, "C1": C1, "C2": C2,
            "C0_int": c0_int,
            "C1_int": c1_int,
            "C2_int": c2_int,
            "err": err
        })

        # 仅打印头部和尾部数据用于观察
        if i < 3 or i == 127:
            # 掩码扩展：2^x 在 [0,1) 的值域是 [1, 2)，C0 必然包含一个整数位 '1'
            # 预留 2 bits 给整数位/符号位防止截断溢出
            mask_t = (1 << (t + 2)) - 1
            mask_p = (1 << (p + 2)) - 1
            mask_q = (1 << (q + 2)) - 1

            c0_display = c0_int & mask_t
            c1_display = c1_int & mask_p 
            c2_display = c2_int & mask_q 
            
            print(f"{i:<8} | {c0_display:<12x} | {c1_display:<12x} | {c2_display:<12x} | {err:.2e}")

    # 防止 err 为 0 时 log2 报错
    good_bits = np.abs(np.log2(errmax)) if errmax > 0 else 0
    print("-" * 75)
    print(f"最大绝对误差 (MAE): {errmax:.12e}")
    print(f"等效精度 (Good Bits): {good_bits:.2f} bits\n")
    
    return results


t_width, p_width, q_width = 24, 18, 13
final_table = compute_coeffs_exp2_128(t_width, p_width, q_width)
save_to_files(final_table, t_width+1, p_width+1, q_width+1, "exp2_coeffs.h", "FP32_EXP2_TABLE", 128)


def compute_coeffs_sigmoid_multi_region(t, p, q):
    """
    t: C0 的小数位宽
    p: C1 的小数位宽
    q: C2 的小数位宽
    针对 Sigmoid 在 [0, 16) 区间的多区间非均匀分段优化 (共 160 段)
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # 定义各个区间：(起点, 终点, 段数, 区间名称)
    regions = [
        (0.0, 1.0, 32, "exp < 0  [0, 1)"),
        (1.0, 2.0, 16, "exp == 0 [1, 2)"),
        (2.0, 4.0, 32, "exp == 1 [2, 4)"),
        (4.0, 8.0, 64, "exp == 2 [4, 8)"),
        (8.0, 16.0, 32,"exp == 3 [8, 16)")
    ]

    errmax = 0
    results = []
    global_seg = 0

    print(f"{'Segment':<8} | {'Region':<16} | {'C0 (Hex)':<12} | {'C1 (Hex)':<12} | {'C2 (Hex)':<12} | {'Error':<10}")
    print("-" * 85)

    for start, end, num_segments, name in regions:
        dx_max = (end - start) / num_segments
        
        for i in range(num_segments):
            # 1. 确定当前段的【中心点】
            m_center = start + i * dx_max + (dx_max / 2.0)
            
            # 2. 采样范围：围绕中心点的相对偏移量 delta
            n = 2**16 
            delta_nodes = np.linspace(-dx_max / 2.0, dx_max / 2.0, n)
            
            # 3. 计算真实的 Y 值
            y_nodes = sigmoid(m_center + delta_nodes)

            # 4. 初始拟合 (二阶多项式)
            poly_coeffs = np.polyfit(delta_nodes, y_nodes, 2)
            a2_raw, a1_raw = poly_coeffs[0], poly_coeffs[1]

            # 5. 独立量化 C1 和 C2
            # Sigmoid 的一阶导数恒为正 (C1 > 0)
            # Sigmoid 在 x>0 时的二阶导数恒为负 (C2 < 0)，转为定点数时会自动按补码保存
            C1 = np.round(a1_raw * (2**p)) * (2**-p)
            C2 = np.round(a2_raw * (2**q)) * (2**-q) 

            # 6. 平衡常数项 C0 (Minimax 平移)
            rem_y = sigmoid(m_center + delta_nodes) - (C1 * delta_nodes + C2 * (delta_nodes**2))
            a0_minimax = (np.max(rem_y) + np.min(rem_y)) / 2.0
            C0 = np.round(a0_minimax * (2**t)) * (2**-t)

            # 7. 误差计算
            test_delta = np.linspace(-dx_max / 2.0, dx_max / 2.0, 500)
            actual_y = sigmoid(m_center + test_delta)
            approx_y = C0 + C1 * test_delta + C2 * (test_delta**2)
            err = np.max(np.abs(actual_y - approx_y))

            if err > errmax:
                errmax = err
            
            c0_int = int(round(C0 * 2**t))
            c1_int = int(round(C1 * 2**p))
            c2_int = int(round(C2 * 2**q))

            results.append({
                "global_seg": global_seg,
                "region": name,
                "C0": C0, "C1": C1, "C2": C2,
                "C0_int": c0_int, "C1_int": c1_int, "C2_int": c2_int,
                "err": err
            })

            # 打印每个区间的头尾部分以便于观察
            if i < 2 or i == num_segments - 1:
                # 掩码扩展以处理符号位和整数位
                mask_t = (1 << (t + 2)) - 1
                mask_p = (1 << (p + 2)) - 1
                mask_q = (1 << (q + 2)) - 1

                c0_display = c0_int & mask_t
                c1_display = c1_int & mask_p 
                c2_display = c2_int & mask_q 
                
                print(f"{global_seg:<8} | {name:<16} | {c0_display:<12x} | {c1_display:<12x} | {c2_display:<12x} | {err:.2e}")
            
            global_seg += 1
        
        # 打印区间分割线
        print("-" * 85)

    good_bits = np.abs(np.log2(errmax)) if errmax > 0 else 0
    print(f"总计分段数: {global_seg}")
    print(f"最大绝对误差 (MAE): {errmax:.12e}")
    print(f"等效精度 (Good Bits): {good_bits:.2f} bits\n")
    
    return results

t_width, p_width, q_width = 27, 18, 13
final_table = compute_coeffs_exp2_128(t_width, p_width, q_width)
save_to_files(final_table, t_width+1, p_width+1, q_width+1, "sig_coeffs.h", "FP32_SIG_TABLE", 176)