# -*- coding: utf-8 -*-
"""
BP神经网络——渗氮层厚度预测（含“产品类型_合格区间_中值”导出模板）
特性：
- 训练：80/20 划分；双隐层 [64,32]+ReLU；Adam(lr=1e-3, wd=1e-5)+CosineAnnealingLR；早停=200
- 图片：训练/验证曲线；验证散点(R$^2$)；±10%灵敏度；±20%因素变化曲线（物理量）
- 工件输出（供 GA 直接引用）：pth / scalers.pkl / best_config_final.json
- 新增输出：产品类型及其合格区间与中值（CSV+JSON），以及逐行中值标注模板
"""

import os, json, time, random, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
import matplotlib.pyplot as plt

# ---------- 显示中文与负号 ----------
matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------- 路径 ----------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "xunlian1208.xlsx")   # 放同目录即可
OUT_FIGS = os.path.join(BASE_DIR, "figs_final")
OUT_DIR  = os.path.join(BASE_DIR, "bp_artifacts")
os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

# ---------- 设备 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] 使用设备：{DEVICE}")

# ---------- 随机种子 ----------
SEED=2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ---------- 列名模糊匹配表 ----------
CAND = {
    "产品类型": ["产品类型","类型","规格","产品","型号","Type","Spec","Category"],
    "挂重量"  : ["挂重量","挂重","重量","Weight"],
    "实际时长": ["实际时长","时长","时间","保温时间","Duration","HoldingTime"],
    "平均加盐量":["平均加盐量","加盐量","盐量","Salt","AvgSalt"],
    "通气量"  : ["通气量","气量","流量","Flow","AirFlow"],
    "平均温度": ["平均温度","温度","Temperature","AvgTemp"],
    "渗氮层厚度(μm)":["渗氮层厚度(μm)","渗氮层厚度","厚度","CaseDepth","Depth"],
    "厚度下限": ["渗层厚度下限","厚度下限","下限","Lower","MinDepth"],
    "厚度上限": ["渗层厚度上限","厚度上限","上限","Upper","MaxDepth"],
}

def fuzzy_pick(cols, cand_list):
    for pat in cand_list:
        for c in cols:
            if pat.lower() in str(c).lower():
                return c
    return None

# ---------- 读取数据 ----------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"未找到数据文件：{DATA_PATH}")
df0 = pd.read_excel(DATA_PATH)
cols = list(df0.columns)

# 必需列
need = ["挂重量","实际时长","平均加盐量","通气量","平均温度","渗氮层厚度(μm)"]
sel = {k: fuzzy_pick(cols, CAND[k]) for k in need}
missing = [k for k,v in sel.items() if v is None]
if missing:
    raise RuntimeError(f"缺少必要列：{missing}；请在表中添加或更正列名（支持模糊匹配）。")

# 可选列（产品类型、上下限）
type_col = fuzzy_pick(cols, CAND["产品类型"])
lower_col = fuzzy_pick(cols, CAND["厚度下限"])
upper_col = fuzzy_pick(cols, CAND["厚度上限"])
if type_col is None:
    raise RuntimeError("缺少‘产品类型’列（如：产品类型/规格/型号/Type/Spec），请在表中提供。")
if (lower_col is None) or (upper_col is None):
    raise RuntimeError("缺少‘渗层厚度下限/上限’列，请在表中提供（支持 模糊匹配：下限/上限/Lower/Upper ...）。")

# 清洗并取数（训练仍然使用数据里实际出现的样本，当前主要是 A 型）
use_cols = [type_col, sel["挂重量"], sel["实际时长"], sel["平均加盐量"],
            sel["通气量"], sel["平均温度"], sel["渗氮层厚度(μm)"], lower_col, upper_col]
df = df0[use_cols].copy()
df.columns = ["产品类型","挂重量","实际时长","平均加盐量","通气量","平均温度","渗氮层厚度(μm)","厚度下限","厚度上限"]
# 数值化
for c in ["挂重量","实际时长","平均加盐量","通气量","平均温度","渗氮层厚度(μm)","厚度下限","厚度上限"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["挂重量","实际时长","平均加盐量","通气量","平均温度","渗氮层厚度(μm)","厚度下限","厚度上限"])
df = df.reset_index(drop=True)
print(f"[Info] 有效样本数：{len(df)}")

# ---------- 生成“逐行中值标注模板”（仍然基于训练数据本身，一般为 A 型） ----------
df_row_mid = df[["产品类型","厚度下限","厚度上限"]].copy()
df_row_mid["合格中值"] = (df_row_mid["厚度下限"] + df_row_mid["厚度上限"]) / 2.0
df_row_mid.to_csv(os.path.join(OUT_DIR,"row_level_mid.csv"), index=False, encoding="utf-8-sig")

# ---------- 计算“产品类型级合格区间与中值”（供 GA/MES 用） ----------
# 这里不再用训练数据 groupby，而是“按工艺表写死所有产品类型的厚度区间”
# 根据你给的工艺表（截图），手工录入各产品类型的渗层厚度范围：
# A: 7~12
# B: 12~20
# C: 12~20
# D: 12~20
# E: 8~12
# F: 8~12
# G: 6~10
# H: 15~18
# I: 15~20
# J: 4~7
# K: 6~8
# L: 8~12
# M: 12~20
# N: 10~14
# O: 12~20
thickness_specs = {
    "A": (7, 12),
    "B": (12, 20),
    "C": (12, 20),
    "D": (12, 20),
    "E": (8, 12),
    "F": (8, 12),
    "G": (6, 10),
    "H": (15, 18),
    "I": (15, 20),
    "J": (4, 7),
    "K": (6, 8),
    "L": (8, 12),
    "M": (12, 20),
    "N": (10, 14),
    "O": (12, 20),
}

# 如果后续工艺表有调整，只需要改上面的数值，不影响训练流程
pi_types = []
pi_low = []
pi_high = []
for ptype, (lo, hi) in thickness_specs.items():
    pi_types.append(ptype)
    pi_low.append(float(lo))
    pi_high.append(float(hi))

grouped = pd.DataFrame({
    "产品类型": pi_types,
    "厚度下限": pi_low,
    "厚度上限": pi_high,
})
grouped["合格中值"] = (grouped["厚度下限"] + grouped["厚度上限"]) / 2.0

grouped.to_csv(os.path.join(OUT_DIR,"product_intervals.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT_DIR,"product_intervals.json"), "w", encoding="utf-8") as f:
    json.dump(grouped.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

# ---------- BP 训练数据 ----------
X_raw = df[["挂重量","实际时长","平均加盐量","通气量","平均温度"]].values.astype(np.float32)
y_raw = df[["渗氮层厚度(μm)"]].values.astype(np.float32)

x_sc, y_sc = MinMaxScaler(), MinMaxScaler()
X = x_sc.fit_transform(X_raw)
y = y_sc.fit_transform(y_raw)

class DS(torch.utils.data.Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i]

full = DS(X,y)
N=len(full); n_tr=int(0.8*N); n_va=N-n_tr
tr, va = random_split(full,[n_tr,n_va], generator=torch.Generator().manual_seed(SEED))
B = min(64, max(8, n_tr//10))
trL = DataLoader(tr, batch_size=B, shuffle=True)
vaL = DataLoader(va, batch_size=B)

# ---------- 模型 ----------
class BP2(nn.Module):
    def __init__(self,h1=64,h2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,h1), nn.ReLU(),
            nn.Linear(h1,h2), nn.ReLU(),
            nn.Linear(h2,1)
        )
    def forward(self,x): return self.net(x)

def train_model(lr=1e-3, wd=1e-5, epochs=2000, patience=200):
    model = BP2().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    loss_fn = nn.MSELoss()
    best=float("inf"); best_state=None; noimp=0
    hist={"train":[], "val":[]}
    for ep in range(epochs):
        model.train(); tl=[]
        for xb,yb in trL:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); l=loss_fn(model(xb), yb); l.backward(); opt.step(); tl.append(l.item())
        model.eval(); vl=[]
        with torch.no_grad():
            for xb,yb in vaL:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                vl.append(loss_fn(model(xb), yb).item())
        trm,vm = float(np.mean(tl)), float(np.mean(vl))
        hist["train"].append(trm); hist["val"].append(vm)
        sch.step()
        if vm < best-1e-9:
            best=vm; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; noimp=0
        else:
            noimp+=1
            if noimp>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, hist, best

def denorm_y(y):
    if isinstance(y, torch.Tensor): y=y.detach().cpu().numpy()
    return y_sc.inverse_transform(y)

def metrics_on_loader(model, loader):
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(DEVICE)
            preds.append(model(xb).cpu()); trues.append(yb.cpu())
    yp = torch.cat(preds,0).numpy(); yt = torch.cat(trues,0).numpy()
    yt_d = denorm_y(yt); yp_d = denorm_y(yp)
    mse=float(mean_squared_error(yt_d, yp_d))
    mae=float(mean_absolute_error(yt_d, yp_d))
    mape=float(np.mean(np.abs((yt_d-yp_d)/np.clip(np.abs(yt_d),1e-8,None)))*100)
    r2=float(r2_score(yt_d, yp_d))
    return {"MSE":mse,"MAE":mae,"MAPE(%)":mape,"R2":r2}, yt_d, yp_d

# ---------- 训练 ----------
model, hist, best_val = train_model()
va_metrics, yt_d, yp_d = metrics_on_loader(model, vaL)
print("[验证集指标]", va_metrics)

# ---------- 图片（保持不变的四类） ----------
# 1) 训练-验证曲线
plt.figure(figsize=(7,5))
plt.plot(hist["train"], label="训练MSE")
plt.plot(hist["val"],   label="验证MSE")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("训练与验证误差变化")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS,"train_val_curve.png"), dpi=180); plt.close()

# 2) 验证集预测散点
plt.figure(figsize=(6,6))
plt.scatter(yt_d, yp_d, s=20, alpha=0.75)
mn, mx = float(min(yt_d.min(), yp_d.min())), float(max(yt_d.max(), yp_d.max()))
plt.plot([mn,mx],[mn,mx],"--")
plt.xlabel("真实厚度 / μm"); plt.ylabel("预测厚度 / μm")
plt.title(f"验证集预测结果 (R$^2$={va_metrics['R2']:.3f}, MSE={va_metrics['MSE']:.3f})")
plt.tight_layout(); plt.savefig(os.path.join(OUT_FIGS,"pred_val.png"), dpi=180); plt.close()

# 3) 参数灵敏度（±10%）
def sensitivity(model, Xn, delta=0.1):
    model.eval()
    base = model(torch.tensor(Xn, dtype=torch.float32, device=DEVICE)).cpu().detach().numpy()
    S=[]
    for i in range(Xn.shape[1]):
        Xu = Xn.copy(); Xd = Xn.copy()
        Xu[:,i]=np.clip(Xu[:,i]*(1+delta), 0, 1)
        Xd[:,i]=np.clip(Xd[:,i]*(1-delta), 0, 1)
        with torch.no_grad():
            yu=model(torch.tensor(Xu,dtype=torch.float32,device=DEVICE)).cpu().numpy()
            yd=model(torch.tensor(Xd,dtype=torch.float32,device=DEVICE)).cpu().numpy()
        base_d=denorm_y(base); yu_d=denorm_y(yu); yd_d=denorm_y(yd)
        change=(np.abs(yu_d-base_d)+np.abs(base_d-yd_d))/2
        S.append(float(np.mean(change)))
    return np.array(S)

labels=["挂重量","实际时长","平均加盐量","通气量","平均温度"]
sens = sensitivity(model, X)
plt.figure(figsize=(7,5))
plt.bar(labels, sens)
plt.ylabel("平均输出变化量 |Δ厚度| / μm")
plt.title("各参数对渗氮层厚度的影响（±10%扰动）")
plt.tight_layout(); plt.savefig(os.path.join(OUT_FIGS,"param_sensitivity.png"), dpi=180); plt.close()

# 4) 各因素变化曲线（使用“未归一化±20%”）
mean_real = X_raw.mean(axis=0)
for i,lab in enumerate(labels):
    base = mean_real.copy()
    rng = np.linspace(0.8*base[i], 1.2*base[i], 50)
    X_tmp = np.repeat(base.reshape(1,-1), 50, axis=0)
    X_tmp[:,i] = rng
    Xn = x_sc.transform(X_tmp)
    with torch.no_grad():
        ypred = model(torch.tensor(Xn,dtype=torch.float32,device=DEVICE)).cpu().numpy()
    ypred_d = denorm_y(ypred)
    plt.figure(figsize=(6,4))
    plt.plot(rng, ypred_d)
    plt.xlabel(f"{lab} (原始量)"); plt.ylabel("预测渗氮层厚度 / μm")
    plt.title(f"{lab} 对渗氮层厚度的影响曲线（±20%变化）")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_FIGS, f"factor_curve_{lab}.png"), dpi=180); plt.close()

# ---------- 保存：模型与归一化器 ----------
torch.save(model.state_dict(), os.path.join(OUT_DIR,"best_bp_model_final.pth"))
with open(os.path.join(OUT_DIR,"scalers.pkl"), "wb") as f:
    pickle.dump({
        "x_scaler": x_sc,
        "y_scaler": y_sc,
        "feature_order": ["挂重量","实际时长","平均加盐量","通气量","平均温度"]
    }, f)

# ---------- 保存：配置与指标 ----------
def np2py(o):
    if isinstance(o,dict): return {k:np2py(v) for k,v in o.items()}
    elif isinstance(o,(list,tuple)): return [np2py(v) for v in o]
    elif isinstance(o,(np.floating, np.integer)): return o.item()
    else: return o

cfg = {
    "arch":[64,32],
    "optimizer":"Adam",
    "lr":1e-3,
    "wd":1e-5,
    "scheduler":"CosineAnnealingLR(T_max=200)",
    "split":"80/20",
    "batch_size": int(B),
    "early_stopping_patience": 200,
    "valid_MSE_norm": float(min(hist["val"])),
    "valid_metrics": va_metrics
}
with open(os.path.join(OUT_DIR,"best_config_final.json"), "w", encoding="utf-8") as f:
    json.dump(np2py(cfg), f, ensure_ascii=False, indent=2)

print("\n✅ 训练完成，并已导出 GA/MES 所需模板与模型工件：")
print(f" - 模型权重：{os.path.join(OUT_DIR,'best_bp_model_final.pth')}")
print(f" - 归一化器：{os.path.join(OUT_DIR,'scalers.pkl')}")
print(f" - 配置说明：{os.path.join(OUT_DIR,'best_config_final.json')}")
print(f" - 产品区间：{os.path.join(OUT_DIR,'product_intervals.csv / .json')}")
print(f" - 逐行中值：{os.path.join(OUT_DIR,'row_level_mid.csv')}")
print(f" - 图片目录：{OUT_FIGS}")
