# -*- coding: utf-8 -*-
"""
分组BP神经网络训练脚本 (含训练/验证集划分与多维评估)
文件名: train_model_split.py
功能：
1. 读取 xunlian1208.xlsx 主数据文件。
2. 根据指定的策略将数据分组 (A, BCD, J, M, O)。
3. 划分训练集和验证集 (80% / 20%)。
4. 训练模型并分别计算训练集、验证集以及验证集中各子类型的 R2, MSE, MAE。
5. 自动判断是否存在过拟合及拟合合理性。
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib

# 配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 确保此处文件名与您实际上传的文件名一致
DATA_PATH = os.path.join(BASE_DIR, "xunlian1208.xlsx") 
ARTIFACTS_DIR = os.path.join(BASE_DIR, "bp_artifacts_split")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==============\n[Info] 训练设备: {DEVICE}\n==============")

SEED = 2025
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 定义训练分组策略
TRAIN_GROUPS = {
    "A": ["A"],
    "BCD": ["B", "C", "D"],
    "J": ["J"],
    "M": ["M"],
    "O": ["O"]
}

# 列名模糊匹配字典
CAND_COLS = {
    "产品类型": ["产品类型", "类型", "规格", "Type", "Spec"],
    "挂重量": ["挂重量", "挂重", "Weight"],
    "实际时长": ["实际时长", "时长", "时间", "Duration", "Time"],
    "平均加盐量": ["平均加盐量", "加盐量", "盐量", "Salt"],
    "通气量": ["通气量", "气量", "流量", "Flow", "Air"],
    "平均温度": ["平均温度", "温度", "Temp"],
    "渗氮层厚度": ["渗氮层厚度", "厚度", "Depth", "Thickness"]
}

def fuzzy_pick(cols, cand_list):
    """从列名列表中模糊查找匹配项"""
    for pat in cand_list:
        for c in cols:
            if pat.lower() in str(c).lower():
                return c
    return None

class BPNet(nn.Module):
    def __init__(self, in_features=5):
        super(BPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x): return self.net(x)

def load_and_clean_data(file_path):
    """读取并清洗总数据"""
    if not os.path.exists(file_path):
        # 如果找不到 xlsx，尝试读取同名 csv (兼容性处理)
        csv_path = file_path.replace(".xlsx", ".csv")
        if os.path.exists(csv_path):
            print(f"[Warning] 未找到 .xlsx，尝试读取 .csv: {csv_path}")
            try:
                df = pd.read_csv(csv_path)
            except:
                df = pd.read_csv(csv_path, encoding='gbk')
        else:
            raise FileNotFoundError(f"未找到数据文件: {file_path}")
    else:
        df = pd.read_excel(file_path)

    cols = df.columns.tolist()

    # 映射列名
    col_map = {}
    for key, cands in CAND_COLS.items():
        found = fuzzy_pick(cols, cands)
        if found:
            col_map[key] = found
        else:
            print(f"[Warning] 未找到列: {key}")
            return None

    # 提取所需列并重命名
    df_clean = pd.DataFrame()
    for std_name, orig_name in col_map.items():
        df_clean[std_name] = df[orig_name]

    # 强制转数值 (除了产品类型)
    num_cols = ["挂重量", "实际时长", "平均加盐量", "通气量", "平均温度", "渗氮层厚度"]
    for c in num_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

    df_clean.dropna(inplace=True)
    # 统一产品类型为大写字符串
    df_clean["产品类型"] = df_clean["产品类型"].astype(str).str.strip().str.upper()

    return df_clean

def calculate_metrics(y_true, y_pred):
    """计算 R2, MSE, MAE"""
    if len(y_true) < 2:
        return {"R2": 0.0, "MSE": 0.0, "MAE": 0.0}
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"R2": round(r2, 4), "MSE": round(mse, 4), "MAE": round(mae, 4)}

def analyze_fit_quality(metrics_train, metrics_val):
    """分析拟合状态和合理性"""
    r2_train = metrics_train["R2"]
    r2_val = metrics_val["R2"]
    mse_train = metrics_train["MSE"]
    mse_val = metrics_val["MSE"]

    analysis = []
    
    # 1. 合理性判断
    if r2_val > 0.7:
        analysis.append("拟合优秀 (Reasonable - Good)")
    elif r2_val > 0.4:
        analysis.append("拟合一般 (Reasonable - Fair)")
    else:
        analysis.append("拟合较差 (Unreasonable - Poor Fit)")

    # 2. 过拟合判断
    # 逻辑：如果训练集 R2 很高，但验证集 R2 明显低 (差距 > 0.15)，或者 验证集 MSE 远大于 训练集 MSE
    if r2_train > 0.5 and (r2_train - r2_val) > 0.15:
        analysis.append("存在过拟合 (Overfitting detected)")
    elif r2_train < 0.3 and r2_val < 0.3:
        analysis.append("欠拟合 (Underfitting)")
    elif r2_val > r2_train:
        analysis.append("验证集表现优于训练集 (Good Generalization or Data Distribution Mismatch)")
    else:
        analysis.append("泛化能力正常 (Normal Generalization)")

    return " | ".join(analysis)

def train_group_split(group_name, target_types, df_all, output_dir):
    """训练单个模型组 (含 Split 和 详细评估)"""
    print(f"\n>>> 开始训练组: {group_name} (包含类型: {target_types})")

    # 筛选数据
    df_group = df_all[df_all["产品类型"].isin(target_types)].copy()

    if len(df_group) < 15: # 稍微提高门槛，因为要切分
        print(f"    [跳过] 样本数不足 ({len(df_group)} 条)")
        return False

    print(f"    [信息] 总样本数: {len(df_group)}")

    feature_cols = ["挂重量", "实际时长", "平均加盐量", "通气量", "平均温度"]
    target_col = "渗氮层厚度"

    # 准备数据
    X_full = df_group[feature_cols].values.astype(np.float32)
    y_full = df_group[[target_col]].values.astype(np.float32)
    types_full = df_group["产品类型"].values

    # 划分训练集和验证集 (尝试分层抽样以保证验证集包含所有类型)
    try:
        X_train, X_val, y_train, y_val, types_train, types_val = train_test_split(
            X_full, y_full, types_full, test_size=0.2, random_state=SEED, stratify=types_full
        )
    except ValueError:
        print("    [提示] 某些类型样本过少，无法进行分层抽样，采用随机划分。")
        X_train, X_val, y_train, y_val, types_train, types_val = train_test_split(
            X_full, y_full, types_full, test_size=0.2, random_state=SEED
        )

    # 归一化 (仅在训练集上Fit，应用到验证集)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    
    X_val_scaled = x_scaler.transform(X_val)
    # y_val 不需要 scale 用于 loss，但评估时需要反归一化预测值，或者我们将 y_val 也 scale 用于计算 loss (可选)
    # 这里我们在训练时不使用 validation loss 做 early stopping，所以只需 scale 用于 model input
    
    # 转换为 Tensor (仅训练集用于梯度更新)
    train_dataset = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train_scaled))
    train_loader = DataLoader(train_dataset, batch_size=min(64, len(X_train)), shuffle=True)

    model = BPNet(in_features=len(feature_cols)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # 训练循环
    epochs = 1500
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()

    # ================= 评估阶段 =================
    model.eval()
    with torch.no_grad():
        # 1. 训练集预测
        p_train_scaled = model(torch.tensor(X_train_scaled, device=DEVICE)).cpu().numpy()
        p_train_raw = y_scaler.inverse_transform(p_train_scaled)
        
        # 2. 验证集预测
        p_val_scaled = model(torch.tensor(X_val_scaled, device=DEVICE)).cpu().numpy()
        p_val_raw = y_scaler.inverse_transform(p_val_scaled)

    # 计算整体指标
    metrics_train = calculate_metrics(y_train, p_train_raw)
    metrics_val = calculate_metrics(y_val, p_val_raw)
    
    # 分析拟合状态
    fit_status = analyze_fit_quality(metrics_train, metrics_val)
    
    print(f"    [训练集] R2: {metrics_train['R2']} | MSE: {metrics_train['MSE']}")
    print(f"    [验证集] R2: {metrics_val['R2']} | MSE: {metrics_val['MSE']}")
    print(f"    [状态] {fit_status}")

    # 计算验证集中 各子类型的指标 (区分不同类型)
    val_type_metrics = {}
    unique_types = np.unique(types_val)
    for t in unique_types:
        # 找到该类型的索引
        mask = (types_val == t)
        if np.sum(mask) > 0:
            y_sub = y_val[mask]
            p_sub = p_val_raw[mask]
            val_type_metrics[str(t)] = calculate_metrics(y_sub, p_sub)

    # ================= 绘图 =================
    plt.figure(figsize=(7, 6))
    
    # 绘制训练集 (蓝色)
    plt.scatter(y_train, p_train_raw, c='blue', alpha=0.4, label='Train Data', marker='o', s=30)
    # 绘制验证集 (橙色)
    plt.scatter(y_val, p_val_raw, c='orange', alpha=0.8, label='Validation Data', marker='^', s=50)
    
    # 理想线
    all_y = np.concatenate([y_train, y_val])
    all_p = np.concatenate([p_train_raw, p_val_raw])
    mn, mx = min(all_y.min(), all_p.min()), max(all_y.max(), all_p.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='Ideal Line')
    
    plt.title(f"Group {group_name} ({','.join(target_types)})\nTrain R2={metrics_train['R2']} | Val R2={metrics_val['R2']}")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fit_plot.png"))
    plt.close()

    # ================= 保存 =================
    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    with open(os.path.join(output_dir, "scalers.pkl"), "wb") as f:
        pickle.dump({
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "feature_order": feature_cols
        }, f)

    # 保存详细指标
    full_metrics = {
        "group": group_name,
        "included_types": target_types,
        "sample_counts": {
            "total": len(df_group),
            "train": len(X_train),
            "val": len(X_val)
        },
        "overall_metrics": {
            "train": metrics_train,
            "validation": metrics_val
        },
        "validation_breakdown_by_type": val_type_metrics,
        "fit_analysis": fit_status
    }
    
    with open(os.path.join(output_dir, "metrics_detailed.json"), "w", encoding='utf-8') as f:
        json.dump(full_metrics, f, indent=2, ensure_ascii=False)

    return True

def main():
    # 1. 读取并清洗数据
    print(f"读取数据文件: {DATA_PATH}")
    try:
        df_all = load_and_clean_data(DATA_PATH)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    if df_all is None:
        print("数据清洗后为空，终止。")
        return

    # 2. 遍历分组进行训练
    product_to_group = {}
    available_groups = []

    for group_name, types in TRAIN_GROUPS.items():
        save_dir = os.path.join(ARTIFACTS_DIR, group_name)
        os.makedirs(save_dir, exist_ok=True)

        success = train_group_split(group_name, types, df_all, save_dir)

        if success:
            available_groups.append(group_name)
            for t in types:
                product_to_group[t] = group_name

    # 3. 生成注册表
    registry = {
        "product_to_group": product_to_group,
        "available_groups": available_groups,
        "description": "产品类型 -> 模型组映射 (Split Version)"
    }

    reg_path = os.path.join(ARTIFACTS_DIR, "model_registry.json")
    with open(reg_path, "w", encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"\n========================================")
    print(f"训练全部完成！")
    print(f"结果保存在: {ARTIFACTS_DIR}")
    print(f"========================================")

if __name__ == "__main__":
    main()