# -*- coding: utf-8 -*-
"""
分组BP神经网络训练优化脚本
文件名: train_model_optimized.py
功能：
1. 数据预处理优化：剔除异常值 (Isolation Forest)。
2. 数据增强：训练集添加高斯噪声 (Data Augmentation)。
3. 模型优化：
   - 引入 Dropout 防止过拟合。
   - 简单的超参数网格搜索 (Grid Search) 寻找最佳 学习率、隐藏层结构、正则化系数。
   - 使用 Early Stopping 防止过拟合。
4. 评估与保存：
   - 详细的过拟合检测。
   - 保存最佳模型及参数配置。
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib

# 配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 确保此处文件名与您实际上传的文件名一致
DATA_PATH = os.path.join(BASE_DIR, "xunlian1208.xlsx")
# 输出目录单独分开
ARTIFACTS_DIR = os.path.join(BASE_DIR, "bp_artifacts_optimized")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==============\n[Info] 训练设备: {DEVICE}\n==============")

SEED = 2025
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 分组策略 (保持不变)
TRAIN_GROUPS = {
    "A": ["A"],
    "BCD": ["B", "C", "D"],
    "J": ["J"],
    "M": ["M"],
    "O": ["O"]
}

# 列名映射
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
    for pat in cand_list:
        for c in cols:
            if pat.lower() in str(c).lower():
                return c
    return None

# ================= 模型定义 (增加 Dropout) =================
class BPNet(nn.Module):
    def __init__(self, in_features, hidden_layers=[64, 32], dropout=0.0):
        super(BPNet, self).__init__()
        layers = []
        last_dim = in_features
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h_dim
        
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ================= 数据处理工具 =================
def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        # 兼容 csv
        csv_path = file_path.replace(".xlsx", ".csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
            except:
                df = pd.read_csv(csv_path, encoding='gbk')
        else:
            raise FileNotFoundError(f"未找到数据文件: {file_path}")
    else:
        df = pd.read_excel(file_path)

    cols = df.columns.tolist()
    col_map = {}
    for key, cands in CAND_COLS.items():
        found = fuzzy_pick(cols, cands)
        if found:
            col_map[key] = found
        else:
            print(f"[Warning] 未找到列: {key}")
            return None

    df_clean = pd.DataFrame()
    for std_name, orig_name in col_map.items():
        df_clean[std_name] = df[orig_name]

    num_cols = ["挂重量", "实际时长", "平均加盐量", "通气量", "平均温度", "渗氮层厚度"]
    for c in num_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

    df_clean.dropna(inplace=True)
    df_clean["产品类型"] = df_clean["产品类型"].astype(str).str.strip().str.upper()
    return df_clean

def remove_outliers_isoforest(X, y, contamination=0.1):
    """使用 Isolation Forest 剔除异常值"""
    if len(X) < 10: return X, y, np.ones(len(X), dtype=bool)
    
    # 组合 X 和 y 一起检测，因为有时候 y 的异常也需要剔除
    data = np.hstack((X, y.reshape(-1, 1)))
    iso = IsolationForest(contamination=contamination, random_state=SEED)
    mask = iso.fit_predict(data) != -1
    return X[mask], y[mask], mask

def augment_data(X, y, noise_scale=0.01, count=1):
    """数据增强：添加微小高斯噪声"""
    X_aug, y_aug = [], []
    # 保留原始数据
    X_aug.append(X)
    y_aug.append(y)
    
    std_devs = np.std(X, axis=0)
    # 防止标准差为0
    std_devs = np.where(std_devs == 0, 1e-5, std_devs)
    
    for _ in range(count):
        # 按特征的标准差比例添加噪声
        noise = np.random.normal(0, noise_scale * std_devs, X.shape)
        X_new = X + noise
        X_aug.append(X_new)
        y_aug.append(y) # y 保持不变
        
    return np.vstack(X_aug), np.vstack(y_aug)

# ================= 训练与搜索流程 =================

def train_one_trial(params, X_train, y_train, X_val, y_val):
    """执行一次超参数训练"""
    # params: {'hidden': [64, 32], 'lr': 0.001, 'dropout': 0.1, 'wd': 1e-4}
    
    model = BPNet(in_features=X_train.shape[1], 
                  hidden_layers=params['hidden'], 
                  dropout=params['dropout']).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    loss_fn = nn.MSELoss()
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    best_val_loss = float('inf')
    best_state = None
    patience = 50
    patience_counter = 0
    
    for epoch in range(1000): # Max epochs
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
            
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # Load best model from this trial
    if best_state:
        model.load_state_dict(best_state)
        
    return model, best_val_loss

def grid_search_train(X_train, y_train, X_val, y_val):
    """超参数网格搜索"""
    # 定义搜索空间：包含简单模型和正则化模型
    search_space = [
        # 方案1: 极简模型，适合数据极少的情况
        {'hidden': [32], 'lr': 0.01, 'dropout': 0.0, 'wd': 1e-4},
        # 方案2: 简单模型，小学习率
        {'hidden': [32, 16], 'lr': 0.005, 'dropout': 0.0, 'wd': 1e-4},
        # 方案3: 标准模型 + 轻微Dropout
        {'hidden': [64, 32], 'lr': 0.001, 'dropout': 0.1, 'wd': 1e-3},
        # 方案4: 标准模型 + 强Dropout (抗过拟合)
        {'hidden': [64, 32], 'lr': 0.001, 'dropout': 0.2, 'wd': 1e-3},
        # 方案5: 较深模型 + 强正则化
        {'hidden': [128, 64, 32], 'lr': 0.001, 'dropout': 0.2, 'wd': 1e-4},
    ]
    
    best_model = None
    best_loss = float('inf')
    best_params = None
    
    print(f"    [Grid Search] 开始搜索 {len(search_space)} 组参数...")
    
    for i, params in enumerate(search_space):
        # print(f"        尝试参数组 {i+1}: {params}")
        model, val_loss = train_one_trial(params, X_train, y_train, X_val, y_val)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_params = params
            
    print(f"    [Grid Search] 最佳参数: {best_params}, Val Loss: {best_loss:.4f}")
    return best_model, best_params

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"R2": round(r2, 4), "MSE": round(mse, 4), "MAE": round(mae, 4)}

def analyze_fit_status(m_train, m_val):
    """
    判断过拟合/欠拟合
    """
    r2_t = m_train['R2']
    r2_v = m_val['R2']
    
    status = []
    
    # 1. 拟合优度
    if r2_v >= 0.7: status.append("拟合优秀")
    elif r2_v >= 0.5: status.append("拟合良好")
    elif r2_v >= 0.3: status.append("拟合一般")
    else: status.append("拟合较差")
    
    # 2. 过拟合检测
    diff = r2_t - r2_v
    # 如果训练集很好(>0.6)，但验证集差太多(>0.15)，则过拟合
    if r2_t > 0.6 and diff > 0.15:
        status.append("存在过拟合 (Overfitting)")
    # 如果两者都差
    elif r2_t < 0.3 and r2_v < 0.3:
        status.append("可能欠拟合 (Underfitting)")
    # 验证集比训练集还好 (常见于Dropout或数据增强后)
    elif r2_v > r2_t:
        status.append("验证集表现更好 (Good Generalization)")
    else:
        status.append("表现正常")
        
    return ", ".join(status)

def train_group_optimized(group_name, target_types, df_all, output_dir):
    print(f"\n>>> 处理组: {group_name} {target_types}")
    
    # 1. 筛选数据
    df_sub = df_all[df_all["产品类型"].isin(target_types)].copy()
    if len(df_sub) < 10:
        print("    样本过少，跳过")
        return False
        
    feature_cols = ["挂重量", "实际时长", "平均加盐量", "通气量", "平均温度"]
    target_col = "渗氮层厚度"
    
    X = df_sub[feature_cols].values
    y = df_sub[target_col].values
    
    # 2. 剔除异常值
    print(f"    原始样本数: {len(X)}")
    X_clean, y_clean, mask = remove_outliers_isoforest(X, y, contamination=0.1) # 10% 异常率
    print(f"    剔除异常后样本数: {len(X_clean)} (剔除了 {len(X) - len(X_clean)} 条)")
    
    if len(X_clean) < 10:
        print("    剔除异常后样本过少，跳过")
        return False
        
    # 3. 划分数据集
    # 尝试分层
    try:
        types = df_sub["产品类型"].values[mask]
        X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=0.2, random_state=SEED, stratify=types)
    except:
        X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=0.2, random_state=SEED)
        
    # 4. 归一化 (使用 MinMaxScaler，因为神经网络喜欢 0-1)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X_train_s = x_scaler.fit_transform(X_train)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1))
    X_val_s = x_scaler.transform(X_val)
    # y_val 不做 transform，用于直接对比，但训练时需要 transform 后的 y
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1))
    
    # 5. 数据增强 (仅训练集)
    print("    应用数据增强...")
    # noise_scale=0.01 (1%的波动), count=2 (扩充2倍)
    X_train_aug, y_train_aug = augment_data(X_train_s, y_train_s, noise_scale=0.01, count=2)
    print(f"    增强后训练集样本数: {len(X_train_aug)}")
    
    # 6. Grid Search 寻找最佳模型
    best_model, best_params = grid_search_train(X_train_aug, y_train_aug, X_val_s, y_val_s)
    
    # 7. 最终评估
    best_model.eval()
    with torch.no_grad():
        p_train_s = best_model(torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        p_val_s = best_model(torch.tensor(X_val_s, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        
    p_train = y_scaler.inverse_transform(p_train_s).flatten()
    p_val = y_scaler.inverse_transform(p_val_s).flatten()
    
    m_train = calculate_metrics(y_train, p_train)
    m_val = calculate_metrics(y_val, p_val)
    status = analyze_fit_status(m_train, m_val)
    
    print(f"    [结果] Train R2: {m_train['R2']}, Val R2: {m_val['R2']}")
    print(f"    [状态] {status}")
    
    # 8. 绘图
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train, p_train, c='blue', alpha=0.3, label=f'Train (R2={m_train["R2"]})')
    plt.scatter(y_val, p_val, c='orange', alpha=0.8, marker='^', s=60, label=f'Val (R2={m_val["R2"]})')
    mn, mx = min(y_clean.min(), p_train.min()), max(y_clean.max(), p_train.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='Ideal')
    plt.title(f"Optimized: Group {group_name}\n{status}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fit_plot.png"))
    plt.close()
    
    # 9. 保存
    torch.save(best_model.state_dict(), os.path.join(output_dir, "model.pth"))
    with open(os.path.join(output_dir, "scalers.pkl"), "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "feature_order": feature_cols}, f)
        
    info = {
        "group": group_name,
        "metrics": {"train": m_train, "val": m_val},
        "best_params": best_params,
        "fit_status": status,
        "samples": {"train_aug": len(X_train_aug), "val": len(X_val)}
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        
    return True

def main():
    print(">>> 读取数据...")
    try:
        df_all = load_and_clean_data(DATA_PATH)
    except Exception as e:
        print(f"数据读取失败: {e}")
        return
        
    if df_all is None: return
    
    reg_groups = []
    prod_map = {}
    
    for g_name, types in TRAIN_GROUPS.items():
        out_dir = os.path.join(ARTIFACTS_DIR, g_name)
        os.makedirs(out_dir, exist_ok=True)
        if train_group_optimized(g_name, types, df_all, out_dir):
            reg_groups.append(g_name)
            for t in types: prod_map[t] = g_name
            
    # Registry
    reg = {
        "available_groups": reg_groups,
        "product_to_group": prod_map,
        "description": "Optimized BP Models with Outlier Removal & Grid Search"
    }
    with open(os.path.join(ARTIFACTS_DIR, "model_registry.json"), "w", encoding='utf-8') as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)
        
    print(f"\n全部完成！结果保存在: {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()