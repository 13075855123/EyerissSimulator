import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

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
    # 定义卷积层配置 (这里使用较小的配置以加速验证准确性)
    # 验证通过后，你可以改回 ResNet 等大网络的配置
    # ==========================================
    # 输入: Batch=3, Channel=32, Height=28, Width=28
    BATCH = 3
    IN_CHANNELS = 32
    INPUT_H, INPUT_W = 28, 28
    
    # 权重: OutChannel=64, InChannel=32, Kernel=3x3
    OUT_CHANNELS = 64
    KERNEL_H, KERNEL_W = 3, 3

    print(f">>> 2. 生成随机数据...")
    print(f"    Input Shape: ({BATCH}, {IN_CHANNELS}, {INPUT_H}, {INPUT_W})")
    print(f"    Weight Shape: ({OUT_CHANNELS}, {IN_CHANNELS}, {KERNEL_H}, {KERNEL_W})")
    
    # 随机生成 float32 类型的数据 (与 PyTorch 更好地对齐，减小精度误差)
    pictures = np.random.randint(0, 10, (BATCH, IN_CHANNELS, INPUT_H, INPUT_W)).astype(np.float32)
    weights = np.random.randint(0, 5, (OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W)).astype(np.float32)

    # ==========================================
    # 打印 Pass 切分信息 (仅作分析，后续直接走完整前向)
    # ==========================================
    print("\n>>> 3. 分析 Hive.CreatePasses 任务切分...")
    passes = hive.CreatePasses(pictures, weights)
    print(f"    总共生成了 {len(passes)} 个 Pass")
    print(f"    [映射参数] t={hive.t}, r={hive.r}, e={hive.e}")

    # ==========================================
    # PyTorch 计算标准答案
    # ==========================================
    print("\n>>> 4. 使用 PyTorch 计算卷积作为 Ground Truth...")
    tensor_pic = torch.from_numpy(pictures)
    tensor_weight = torch.from_numpy(weights)
    
    # EyerissSimulator 的卷积默认没有 padding，stride 为 1
    torch_out = F.conv2d(tensor_pic, tensor_weight, stride=1, padding=0)
    pytorch_result = torch_out.numpy()
    print(f"    PyTorch 输出形状: {pytorch_result.shape}")

    # ==========================================
    # Eyeriss Simulator (Hive) 计算
    # ==========================================
    print("\n>>> 5. 使用 EyerissSimulator 计算卷积 (这可能需要几秒到几十秒)...")
    
    # 根据 IO2/Hive 逻辑，输入模拟器的数据通常是经过 RLE Compress 的
    comp_pic = hive.RLE.Compress(pictures)
    comp_weight = hive.RLE.Compress(weights)
    
    # 运行模拟器执行卷积并组装 OfMaps
    hive_out_comp = hive.Conv2d(comp_pic, comp_weight)
    
    # 解压输出结果转回 numpy
    hive_result = hive.RLE.Decompress(hive_out_comp)
    print(f"    Hive 输出形状: {hive_result.shape}")

    # ==========================================
    # 对比结果
    # ==========================================
    print("\n>>> 6. 对比结果...")
    if pytorch_result.shape != hive_result.shape:
        print(f"    ❌ 形状不一致! PyTorch: {pytorch_result.shape}, Hive: {hive_result.shape}")
        return

    diff = np.abs(pytorch_result - hive_result)
    max_diff = np.max(diff)
    print(f"    最大绝对误差 (Max Absolute Error): {max_diff}")
    
    if max_diff < 1e-4:
        print("    ✅ 测试通过！Hive 循环重排和 Pass 切分与标准卷积结果完全一致！")
    else:
        print("    ❌ 测试失败！Hive 的计算结果与 PyTorch 不一致。")
        # 打印错误位置帮助 Debug
        err_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    误差最大的位置索引: {err_idx}")
        print(f"    PyTorch 该位置的值: {pytorch_result[err_idx]}")
        print(f"    Hive 该位置的值: {hive_result[err_idx]}")

if __name__ == "__main__":
    run_pass_analysis()