import numpy as np
import os
import sys

# 获取当前脚本所在目录 (test文件夹)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上退一级，获取项目根目录 (EyerissSimulator文件夹)
project_root = os.path.dirname(current_dir)

# 将项目根目录加入到 Python 的搜索路径中
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.EyerissF import EyerissF
from src.Hive import Hive
from src import conf

def run_pass_analysis():

    # --- 强行覆盖默认的缓存配置 ---
    conf.IfmapSpad = 3   # 迫使 q = 3
    conf.FilterSpad = 3  # 迫使 p = 3
    conf.PsumSpad = 1
    # -----------------------------

    print(">>> 1. 初始化模拟环境...")
    eyeriss = EyerissF()
    hive = Hive(eyeriss)

    # ==========================================
    # 定义一个简单的卷积层配置 (模拟 ResNet 的某一层)
    # ==========================================
    # 输入: Batch=3, Channel=32, Height=28, Width=28
    BATCH = 3
    IN_CHANNELS = 32
    INPUT_H, INPUT_W = 28, 28
    
    # 权重: OutChannel=64, InChannel=32, Kernel=3x3
    OUT_CHANNELS = 64
    KERNEL_H, KERNEL_W = 3, 3
    
    # 🌟 新增：定义测试的步长
    STRIDE = 2

    print(f">>> 2. 生成随机数据...")
    print(f"    Input Shape: ({BATCH}, {IN_CHANNELS}, {INPUT_H}, {INPUT_W})")
    print(f"    Weight Shape: ({OUT_CHANNELS}, {IN_CHANNELS}, {KERNEL_H}, {KERNEL_W})")
    print(f"    Stride: {STRIDE}")
    
    # 随机生成 uint8 类型的图像数据和权重数据
    pictures = np.random.randint(0, 10, (BATCH, IN_CHANNELS, INPUT_H, INPUT_W))
    weights = np.random.randint(0, 5, (OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W))

    # ==========================================
    # 执行 Mapping 并生成 Passes
    # ==========================================
    print("\n>>> 3. 调用 Hive.CreatePasses 进行任务切分...")
    
    # 🌟 新增：在执行切分前，告诉调度器当前的步长
    hive.stride = STRIDE
    # 这一步会自动计算 Mapping 参数 (m, n, p, q, r, t) 并切分数据
    passes = hive.CreatePasses(pictures, weights)
    
    print(f"\n[结果] 总共生成了 {len(passes)} 个 Pass")
    print(f"[映射参数] Hive 自动计算的 Mapping 参数:")
    print(f"    t (Filter并行数/12行PE阵列卷积核并行数): {hive.t}")
    print(f"    r (Channel并行数): {hive.r}")
    print(f"    e (本次Pass负责的输出特征图行数):   {hive.e}")

    # ==========================================
    # 4. 分析并保存前几个 Pass
    # ==========================================
    print("\n>>> 4. 提取并保存前几个 Pass 供分析...")
    save_dir = "saved_passes"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    num_to_save = 3 # 保存前3个作为示例
    
    for i in range(min(num_to_save, len(passes))):
        print(f"\n--- 分析并保存 Pass {i} ---")
        pic_pass, weight_pass = passes[i]
        
        print(f"    PicPass Shape:    {pic_pass.shape}")
        print(f"    WeightPass Shape: {weight_pass.shape}")
        
        # 简单的数据统计，用于验证
        print(f"    PicPass 数据量:    {pic_pass.size} 元素")
        print(f"    WeightPass 数据量: {weight_pass.size} 元素")
        
        # 保存文件
        pic_filename = os.path.join(save_dir, f"pass_{i}_pic.npy")
        weight_filename = os.path.join(save_dir, f"pass_{i}_weight.npy")
        
        np.save(pic_filename, pic_pass)
        np.save(weight_filename, weight_pass)
        print(f"    已保存到: {pic_filename} 和 {weight_filename}")

    print(f"\n>>> 运行完成。请检查 '{save_dir}' 文件夹查看保存的 .npy 文件。")

if __name__ == "__main__":
    run_pass_analysis()