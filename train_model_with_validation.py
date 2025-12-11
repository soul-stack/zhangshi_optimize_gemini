# -*- coding: utf-8 -*-
"""
分组BP神经网络训练脚本 (带训练/验证集划分)
功能：
1. 读取 xunlian1208.xlsx 主数据文件。
2. 根据指定策略将数据分组 (A, BCD, J, M, O)。
3. 在每个分组内进行训练/验证划分，分别输出 R2/MSE/MAE。
4. 给出过拟合与拟合合理性的简单判断。
5. 生成 model_registry.json 供 API 服务调用。
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib

# 配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 修改：指定使用 xunlian1208.xlsx
DATA_PATH = os.path.join(BASE_DIR, "xunlian1208.xlsx")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "bp_artifacts_val")
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
# 键为模型组名称，值为包含的产品类型列表
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
        # 增加网络容量以适应可能的复杂数据
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def load_and_clean_data(file_path):
    """读取并清洗总数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据文件: {file_path}")

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
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

    df_clean.dropna(inplace=True)
    # 统一产品类型为大写字符串
    df_clean["产品类型"] = df_clean["产品类型"].astype(str).str.strip().str.upper()

    return df_clean


def assess_fit(train_metrics, val_metrics):
    """根据训练/验证指标给出简单的过拟合与拟合合理性判断"""
    train_r2, val_r2 = train_metrics["R2"], val_metrics["R2"]
    train_mse, val_mse = train_metrics["MSE"], val_metrics["MSE"]

    overfit = (train_r2 - val_r2 > 0.1) and (val_mse > train_mse * 1.2)
    reasonable_fit = val_r2 >= 0.5 and val_mse <= train_mse * 3

    if overfit:
        note = "验证集表现明显弱于训练集，存在过拟合风险。"
    elif not reasonable_fit:
        note = "验证集拟合度较低，模型拟合可能不足，需要调整特征或参数。"
    else:
        note = "训练与验证指标接近，拟合较为合理。"

    return {"overfit": overfit, "reasonable_fit": reasonable_fit, "note": note}


def train_group(group_name, target_types, df_all, output_dir):
    """训练单个模型组，包含训练/验证划分与指标输出"""
    print(f"\n>>> 开始训练组: {group_name} (包含类型: {target_types})")

    # 筛选数据
    df_group = df_all[df_all["产品类型"].isin(target_types)].copy()

    if len(df_group) < 12:
        print(f"    [跳过] 样本数不足 ({len(df_group)} 条，至少需要12条用于划分训练/验证)")
        return False

    print(f"    [信息] 样本数: {len(df_group)}")

    # 准备数据
    feature_cols = ["挂重量", "实际时长", "平均加盐量", "通气量", "平均温度"]
    target_col = "渗氮层厚度"

    X_raw = df_group[feature_cols].values.astype(np.float32)
    y_raw = df_group[[target_col]].values.astype(np.float32)

    # 划分训练/验证集
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=SEED, shuffle=True
    )

    # 归一化：仅用训练集拟合
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    y_train = y_scaler.fit_transform(y_train_raw)

    X_val = x_scaler.transform(X_val_raw)
    y_val = y_scaler.transform(y_val_raw)

    # 转换为 Tensor
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=min(64, len(train_ds)), shuffle=True)

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

        if (epoch + 1) % 500 == 0:
            print(f"    Epoch {epoch + 1}/{epochs} Loss: {loss.item():.6f}")

    # 评估
    model.eval()
    with torch.no_grad():
        pred_train_scaled = model(torch.tensor(X_train, device=DEVICE)).cpu().numpy()
        pred_val_scaled = model(torch.tensor(X_val, device=DEVICE)).cpu().numpy()

    pred_train = y_scaler.inverse_transform(pred_train_scaled)
    pred_val = y_scaler.inverse_transform(pred_val_scaled)

    # 还原真实值
    y_train_real = y_train_raw
    y_val_real = y_val_raw

    metrics_train = {
        "R2": r2_score(y_train_real, pred_train),
        "MSE": mean_squared_error(y_train_real, pred_train),
        "MAE": mean_absolute_error(y_train_real, pred_train),
        "Samples": len(y_train_real),
    }

    metrics_val = {
        "R2": r2_score(y_val_real, pred_val),
        "MSE": mean_squared_error(y_val_real, pred_val),
        "MAE": mean_absolute_error(y_val_real, pred_val),
        "Samples": len(y_val_real),
    }

    assessment = assess_fit(metrics_train, metrics_val)

    print(
        "    [训练集] R2: {R2:.4f} | MSE: {MSE:.4f} | MAE: {MAE:.4f} (样本 {Samples})".format(
            **metrics_train
        )
    )
    print(
        "    [验证集] R2: {R2:.4f} | MSE: {MSE:.4f} | MAE: {MAE:.4f} (样本 {Samples})".format(
            **metrics_val
        )
    )
    print(f"    [判断] {assessment['note']}")

    # 保存图表
    plt.figure(figsize=(7, 6))
    plt.scatter(y_train_real, pred_train, alpha=0.6, label="Train")
    plt.scatter(y_val_real, pred_val, alpha=0.8, label="Val", marker="^")
    mn = min(y_train_real.min(), y_val_real.min(), pred_train.min(), pred_val.min())
    mx = max(y_train_real.max(), y_val_real.max(), pred_train.max(), pred_val.max())
    plt.plot([mn, mx], [mn, mx], "r--", label="Ideal")
    plt.title(
        f"Group {group_name} ({','.join(target_types)})\n"
        f"Train R2={metrics_train['R2']:.3f} | Val R2={metrics_val['R2']:.3f}"
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fit_plot.png"))
    plt.close()

    # 保存模型和Scaler
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    with open(os.path.join(output_dir, "scalers.pkl"), "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "feature_order": feature_cols}, f)

    # 保存指标
    metrics = {
        "train": metrics_train,
        "val": metrics_val,
        "assessment": assessment,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return True


def main():
    # 1. 读取并清洗数据
    print(f"读取数据文件: {DATA_PATH}")
    df_all = load_and_clean_data(DATA_PATH)
    if df_all is None:
        print("数据读取失败，终止。")
        return

    # 2. 遍历分组进行训练
    product_to_group = {}
    available_groups = []

    for group_name, types in TRAIN_GROUPS.items():
        save_dir = os.path.join(ARTIFACTS_DIR, group_name)
        os.makedirs(save_dir, exist_ok=True)

        success = train_group(group_name, types, df_all, save_dir)

        if success:
            available_groups.append(group_name)
            for t in types:
                product_to_group[t] = group_name

    # 3. 生成注册表
    registry = {
        "product_to_group": product_to_group,
        "available_groups": available_groups,
        "description": "产品类型 -> 模型组映射 (基于 xunlian1208，含训练/验证指标)",
    }

    reg_path = os.path.join(ARTIFACTS_DIR, "model_registry.json")
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print("\n========================================")
    print("训练全部完成！")
    print(f"成功训练的组: {available_groups}")
    print(f"映射表已保存: {reg_path}")
    print("========================================")


if __name__ == "__main__":
    main()
