import numpy as np
import os

# 这里替换成你想查看的文件路径
# 比如查看 filter 文件夹下的 convnet.c1.weight.npy
file_path = 'saved_passes\pass_1_pic.npy' 

if os.path.exists(file_path):
    # 加载数据
    data = np.load(file_path)
    
    print(f"=== 文件: {file_path} ===")
    print(f"数据类型 (dtype): {data.dtype}")
    print(f"数组形状 (shape): {data.shape}")
    print("-" * 30)
    print("数据内容:")
    np.set_printoptions(threshold=np.inf) # 设置打印阈值为无限大
    print(data)
else:
    print(f"错误: 找不到文件 '{file_path}'，请检查路径。")