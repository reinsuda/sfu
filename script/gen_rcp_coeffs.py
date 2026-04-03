import numpy as np
import csv
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