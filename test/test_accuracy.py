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
    # 定义卷积层配置
    # ==========================================
    # 输入: Batch=3, Channel=32, Height=28, Width=28
    BATCH = 3
    IN_CHANNELS = 32
    INPUT_H, INPUT_W = 28, 28
    
    # 权重: OutChannel=64, InChannel=32, Kernel=3x3
    OUT_CHANNELS = 64
    KERNEL_H, KERNEL_W = 3, 3

    # 定义测试的步长
    STRIDE = 2
    

    print(f">>> 2. 生成随机数据...")
    print(f"    Input Shape: ({BATCH}, {IN_CHANNELS}, {INPUT_H}, {INPUT_W})")
    print(f"    Weight Shape: ({OUT_CHANNELS}, {IN_CHANNELS}, {KERNEL_H}, {KERNEL_W})")
    print(f"    Stride: {STRIDE}")
    
    # 随机生成 float32 类型的数据
    pictures = np.random.randint(0, 10, (BATCH, IN_CHANNELS, INPUT_H, INPUT_W)).astype(np.float32)
    weights = np.random.randint(0, 5, (OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W)).astype(np.float32)

    # ==========================================
    # 打印 Pass 切分信息 (仅作分析展示)
    # ==========================================
    print("\n>>> 3. 分析 Hive.CreatePasses 任务切分...")
    
    # 🌟 新增：手动设置 stride 并调用 CreatePasses 获取切分结果进行展示
    hive.stride = STRIDE
    passes = hive.CreatePasses(pictures, weights)
    
    print(f"    [结果] 总共生成了 {len(passes)} 个 Pass")
    print(f"    [映射参数] Hive 自动计算的 Mapping 参数:")
    print(f"        t (Filter并行数): {hive.t}")
    print(f"        r (Channel并行数): {hive.r}")
    print(f"        e (输入行切片):   {hive.e}")

    # 🌟 新增：展示前几个 Pass 的大小
    num_to_show = min(3, len(passes))
    for i in range(num_to_show):
        pic_pass, weight_pass = passes[i]
        print(f"    --- Pass {i} 形状展示 ---")
        print(f"        PicPass Shape:    {pic_pass.shape}")
        print(f"        WeightPass Shape: {weight_pass.shape}")

    # ==========================================
    # PyTorch 计算标准答案
    # ==========================================
    print("\n>>> 4. 使用 PyTorch 计算卷积作为 Ground Truth...")
    tensor_pic = torch.from_numpy(pictures)
    tensor_weight = torch.from_numpy(weights)
    
    # 在此处传入 STRIDE 变量
    torch_out = F.conv2d(tensor_pic, tensor_weight, stride=STRIDE, padding=0)
    pytorch_result = torch_out.numpy()
    print(f"    PyTorch 输出形状: {pytorch_result.shape}")

    # ==========================================
    # Eyeriss Simulator (Hive) 计算
    # ==========================================
    print("\n>>> 5. 使用 EyerissSimulator 计算卷积 (这可能需要几秒到几十秒)...")
    
    comp_pic = hive.RLE.Compress(pictures)
    comp_weight = hive.RLE.Compress(weights)
    
    # 在此处传入 STRIDE 变量，模拟器内部会再次自动切分并执行
    hive_out_comp = hive.Conv2d(comp_pic, comp_weight, stride=STRIDE)
    
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
        print(f"    ✅ 测试通过！Hive 在 Stride={STRIDE} 下的计算结果与标准卷积完全一致！")
    else:
        print("    ❌ 测试失败！Hive 的计算结果与 PyTorch 不一致。")
        err_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    误差最大的位置索引: {err_idx}")
        print(f"    PyTorch 该位置的值: {pytorch_result[err_idx]}")
        print(f"    Hive 该位置的值: {hive_result[err_idx]}")

if __name__ == "__main__":
    run_pass_analysis()