rom_table = []

for i in range(32):
    # 1. 计算区间的起点
    # 5 bit 表示 32 个区间，每个区间宽度为 1/32 = 0.03125
    start_val = 1.0 + (i / 32.0)
    
    # 2. 取区间中点，以获得最小的平均误差
    mid_val = start_val + (1.0 / 64.0)
    
    # 3. 计算平方
    sq_val = mid_val * mid_val
    
    # 4. 转换为 Q23 格式的整数 (1.0 = 2^23 = 8388608)
    # 因为 M^2 最大接近 4.0，所以需要 25 bit，完美装进 uint32_t
    q23_val = int(round(sq_val * 8388608))
    
    rom_table.append(q23_val)

# 打印 C++ 数组格式
print("const uint32_t M_SQ_ROM[32] = {")
for i in range(0, 32, 4):
    row = ", ".join([f"0x{val:07X}" for val in rom_table[i:i+4]])
    print(f"    {row},")
print("};")